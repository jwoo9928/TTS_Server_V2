import os
import time
import uuid
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import multiprocessing
import logging
from flask import Flask, request, jsonify, send_file
from melo.api import TTS
import psutil
import tempfile

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSServer:
    def __init__(self, language='KR', device='cpu', max_workers=None):
        """
        TTS 서버 초기화
        
        Args:
            language: TTS 모델 언어 설정
            device: 'cpu' 또는 'cuda:0'과 같은 장치 설정
            max_workers: 작업자 스레드 수 (None일 경우 자동 설정)
        """
        self.language = language
        self.device = device
        
        # 사용 가능한 CPU 코어 수 확인
        cpu_count = multiprocessing.cpu_count()
        logger.info(f"사용 가능한 CPU 코어 수: {cpu_count}")
        
        # 작업자 수 설정 (기본값: CPU 코어 수)
        if max_workers is None:
            self.max_workers = cpu_count
        else:
            self.max_workers = max_workers
        
        logger.info(f"작업자 스레드 수: {self.max_workers}")
        
        # 작업 큐 초기화
        self.request_queue = queue.Queue()
        
        # 진행 중인 작업 추적
        self.active_requests = {}
        self.request_lock = threading.Lock()
        
        # 세마포어를 사용하여 동시 작업 수 제한
        self.semaphore = threading.Semaphore(self.max_workers)
        
        # TTS 모델 초기화
        self.initialize_model()
        
        # 처리 스레드 시작
        self.start_worker_threads()
        
        # 성능 모니터링 스레드 시작
        self.start_monitoring_thread()
        
        # 임시 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        logger.info(f"임시 파일 디렉토리 생성: {self.temp_dir}")

    def initialize_model(self):
        """TTS 모델 초기화"""
        logger.info(f"TTS 모델 초기화 중 (언어: {self.language}, 장치: {self.device})...")
        start_time = time.time()
        self.model = TTS(language=self.language, device=self.device)
        self.speaker_ids = self.model.hps.data.spk2id
        logger.info(f"TTS 모델 초기화 완료 (소요 시간: {time.time() - start_time:.2f}초)")
        logger.info(f"사용 가능한 화자 ID: {list(self.speaker_ids.keys())}")

    def start_worker_threads(self):
        """작업자 스레드 풀 시작"""
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        for _ in range(self.max_workers):
            self.executor.submit(self.worker_thread)
        logger.info(f"{self.max_workers}개의 작업자 스레드 시작됨")

    def worker_thread(self):
        """작업 처리 스레드"""
        while True:
            try:
                # 큐에서 작업 가져오기
                request_id, text, speaker_id, speed = self.request_queue.get()
                
                # 세마포어 획득 (동시 실행 제한)
                self.semaphore.acquire()
                
                try:
                    # TTS 처리 실행
                    logger.info(f"요청 처리 중: {request_id[:8]} (텍스트 길이: {len(text)}자, 속도: {speed})")
                    start_time = time.time()
                    
                    output_path = os.path.join(self.temp_dir, f"{request_id}.wav")
                    self.model.tts_to_file(text, self.speaker_ids[speaker_id], output_path, speed=speed)
                    
                    process_time = time.time() - start_time
                    logger.info(f"요청 완료: {request_id[:8]} (처리 시간: {process_time:.2f}초)")
                    
                    # 상태 업데이트
                    with self.request_lock:
                        self.active_requests[request_id] = {
                            'status': 'completed',
                            'file_path': output_path,
                            'processing_time': process_time
                        }
                except Exception as e:
                    logger.error(f"TTS 처리 오류 (요청 ID: {request_id[:8]}): {str(e)}")
                    with self.request_lock:
                        self.active_requests[request_id] = {
                            'status': 'error',
                            'error': str(e)
                        }
                finally:
                    # 세마포어 해제
                    self.semaphore.release()
                    # 큐 작업 완료 표시
                    self.request_queue.task_done()
            except Exception as e:
                logger.error(f"작업자 스레드 오류: {str(e)}")

    def start_monitoring_thread(self):
        """시스템 리소스 모니터링 스레드"""
        def monitor_resources():
            while True:
                try:
                    cpu_percent = psutil.cpu_percent(interval=5)
                    memory_percent = psutil.virtual_memory().percent
                    queue_size = self.request_queue.qsize()
                    
                    logger.info(f"시스템 모니터링 - CPU: {cpu_percent}%, 메모리: {memory_percent}%, 대기 큐: {queue_size}")
                    
                    # 완료된 파일 정리 (5분 이상 지난 파일)
                    self.cleanup_completed_files(300)
                    
                    time.sleep(10)  # 10초마다 업데이트
                except Exception as e:
                    logger.error(f"모니터링 오류: {str(e)}")
                    time.sleep(10)
        
        monitoring_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitoring_thread.start()
        logger.info("리소스 모니터링 스레드 시작됨")

    def cleanup_completed_files(self, max_age=300):
        """처리 완료된 파일 정리"""
        current_time = time.time()
        with self.request_lock:
            for req_id in list(self.active_requests.keys()):
                request_data = self.active_requests[req_id]
                if request_data.get('status') == 'completed':
                    if request_data.get('completion_time', current_time) < current_time - max_age:
                        file_path = request_data.get('file_path')
                        if file_path and os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                                logger.info(f"오래된 파일 삭제: {file_path}")
                            except Exception as e:
                                logger.error(f"파일 삭제 오류: {str(e)}")
                        # 요청 정보 삭제
                        del self.active_requests[req_id]

    def process_tts(self, text, speaker_id=None, speed=1.0):
        """
        TTS 처리 요청
        
        Args:
            text: 변환할 텍스트
            speaker_id: 화자 ID (None일 경우 기본 언어 화자)
            speed: 재생 속도
            
        Returns:
            request_id: 요청 ID
        """
        if not speaker_id:
            speaker_id = self.language
            
        # 요청 ID 생성
        request_id = str(uuid.uuid4())
        
        # 초기 상태 설정
        with self.request_lock:
            self.active_requests[request_id] = {'status': 'processing'}
        
        # 요청 큐에 추가
        self.request_queue.put((request_id, text, speaker_id, speed))
        logger.info(f"요청 접수: {request_id[:8]} (대기 큐 크기: {self.request_queue.qsize()})")
        
        return request_id
    
    def get_request_status(self, request_id):
        """요청 상태 확인"""
        with self.request_lock:
            if request_id in self.active_requests:
                return self.active_requests[request_id]
            return {'status': 'not_found'}

# Flask 앱 정의
app = Flask(__name__)
tts_server = None  # 전역 서버 인스턴스

@app.route('/tts', methods=['POST'])
def request_tts():
    """TTS 변환 요청 API - 동기 처리"""
    try:
        # JSON 또는 폼 데이터에서 텍스트 추출
        if request.is_json:
            data = request.get_json()
            text = data.get('text')
            speaker_id = data.get('speaker_id')
            speed = float(data.get('speed', 1.0))
        else:
            text = request.form.get('text')
            speaker_id = request.form.get('speaker_id')
            speed = float(request.form.get('speed', 1.0))
            
        # 파라미터 검증
        if not text:
            return jsonify({'error': '텍스트가 필요합니다'}), 400
            
        if not speaker_id:
            speaker_id = tts_server.language
            
        # TTS 처리 요청
        request_id = tts_server.process_tts(text, speaker_id, speed)
        
        # 작업 완료될 때까지 대기 (최대 60초)
        max_wait = 60
        wait_time = 0
        wait_interval = 0.5
        
        while wait_time < max_wait:
            status = tts_server.get_request_status(request_id)
            if status['status'] == 'completed':
                # 파일 반환
                return send_file(
                    status['file_path'], 
                    mimetype='audio/wav',
                    as_attachment=True,
                    download_name=f"tts_{request_id[:8]}.wav"
                )
            elif status['status'] == 'error':
                return jsonify({'error': status.get('error', '알 수 없는 오류')}), 500
                
            time.sleep(wait_interval)
            wait_time += wait_interval
            
        # 시간 초과
        return jsonify({
            'request_id': request_id,
            'status': 'processing',
            'message': '처리 시간이 오래 걸립니다. /status 엔드포인트로 상태를 확인하고 /download 엔드포인트로 파일을 다운로드하세요.'
        }), 202
            
    except Exception as e:
        logger.error(f"TTS 요청 처리 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/tts/async', methods=['POST'])
def request_tts_async():
    """TTS 변환 요청 API - 비동기 처리"""
    try:
        # JSON 또는 폼 데이터에서 텍스트 추출
        if request.is_json:
            data = request.get_json()
            text = data.get('text')
            speaker_id = data.get('speaker_id')
            speed = float(data.get('speed', 1.0))
        else:
            text = request.form.get('text')
            speaker_id = request.form.get('speaker_id')
            speed = float(request.form.get('speed', 1.0))
            
        # 파라미터 검증
        if not text:
            return jsonify({'error': '텍스트가 필요합니다'}), 400
            
        if not speaker_id:
            speaker_id = tts_server.language
            
        # TTS 처리 요청
        request_id = tts_server.process_tts(text, speaker_id, speed)
        
        return jsonify({
            'request_id': request_id,
            'status': 'processing',
            'message': '비동기 처리 중입니다. /status 엔드포인트로 상태를 확인하고 /download 엔드포인트로 파일을 다운로드하세요.'
        })
            
    except Exception as e:
        logger.error(f"TTS 요청 처리 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/status/<request_id>', methods=['GET'])
def check_status(request_id):
    """TTS 처리 상태 확인 API"""
    try:
        status = tts_server.get_request_status(request_id)
        return jsonify(status)
    except Exception as e:
        logger.error(f"상태 확인 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<request_id>', methods=['GET'])
def download_file(request_id):
    """TTS 결과 파일 다운로드 API"""
    try:
        status = tts_server.get_request_status(request_id)
        
        if status['status'] == 'completed' and 'file_path' in status:
            return send_file(
                status['file_path'], 
                mimetype='audio/wav',
                as_attachment=True,
                download_name=f"tts_{request_id[:8]}.wav"
            )
        elif status['status'] == 'error':
            return jsonify({'error': status.get('error', '알 수 없는 오류')}), 500
        elif status['status'] == 'processing':
            return jsonify({'status': 'processing', 'message': '처리 중입니다'}), 202
        else:
            return jsonify({'error': '파일을 찾을 수 없습니다'}), 404
    except Exception as e:
        logger.error(f"파일 다운로드 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/speakers', methods=['GET'])
def get_speakers():
    """사용 가능한 화자 목록 API"""
    try:
        return jsonify({
            'speakers': list(tts_server.speaker_ids.keys())
        })
    except Exception as e:
        logger.error(f"화자 목록 조회 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """서버 상태 확인 API"""
    try:
        queue_size = tts_server.request_queue.qsize()
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        return jsonify({
            'status': 'healthy',
            'queue_size': queue_size,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'workers': tts_server.max_workers
        })
    except Exception as e:
        logger.error(f"상태 확인 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

def parse_arguments():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='고성능 TTS 서버')
    parser.add_argument('--port', type=int, default=8080, help='서버 포트 (기본값: 8080)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='서버 호스트 (기본값: 0.0.0.0)')
    parser.add_argument('--language', type=str, default='KR', help='TTS 언어 (기본값: KR)')
    parser.add_argument('--device', type=str, default='cpu', help='장치 설정 (기본값: cpu, 또는 cuda:0)')
    parser.add_argument('--workers', type=int, default=None, help='작업자 스레드 수 (기본값: CPU 코어 수)')
    return parser.parse_args()

def main():
    """메인 함수"""
    args = parse_arguments()
    
    # 전역 서버 인스턴스 초기화
    global tts_server
    tts_server = TTSServer(language=args.language, device=args.device, max_workers=args.workers)
    
    logger.info(f"서버 시작 중 (호스트: {args.host}, 포트: {args.port})...")
    app.run(host=args.host, port=args.port, threaded=True)

if __name__ == '__main__':
    main()