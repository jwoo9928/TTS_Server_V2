// src/tts/tts.service.ts
import { Injectable, OnModuleInit, Logger, InternalServerErrorException, BadRequestException } from '@nestjs/common';
import { KokoroTTS, KokoroVoice } from 'kokoro-js'; // Assuming KokoroVoice type exists

// Define a placeholder type if KokoroTTS doesn't export its instance type explicitly
type KokoroTTSInstance = any; // Replace 'any' if a specific type is available
type KokoroAudioOutput = {
    // Assuming a method like this exists based on common patterns
    // Adjust if the actual API is different (e.g., returns Buffer directly or needs save/read)
    getWavBytes: () => Promise<Buffer> | Buffer;
    save: (path: string) => Promise<void>;
    // Add other potential methods like getSampleRate(), getDuration() etc.
};


@Injectable()
export class TtsService implements OnModuleInit {
    private readonly logger = new Logger(TtsService.name);
    private tts: KokoroTTSInstance | null = null;
    private isModelReady = false;
    private availableVoices: KokoroVoice[] = []; // Assuming KokoroVoice type

    // --- Configuration ---
    // Ideally, load these from config (e.g., @nestjs/config)
    private readonly modelId = process.env.KOKORO_MODEL_ID || 'onnx-community/Kokoro-82M-ONNX';
    private readonly dtype = process.env.KOKORO_DTYPE || 'q8'; // e.g., "fp32", "fp16", "q8", "q4", "q4f16"
    private readonly defaultVoice = process.env.KOKORO_DEFAULT_VOICE || 'en_us_lisa'; // Example default

    async onModuleInit() {
        await this.loadModel();
    }

    private async loadModel(): Promise<void> {
        if (this.isModelReady) return;

        this.logger.log(`Worker ${process.pid}: Loading TTS model "${this.modelId}" with dtype "${this.dtype}"...`);
        try {
            // Ensure KokoroTTS is correctly imported (CommonJS vs ESM might need adjustments)
            // If KokoroTTS is ESM-only and your project is CommonJS, you might need dynamic import:
            // const { KokoroTTS } = await import('kokoro-js');

            this.tts = await KokoroTTS.from_pretrained(this.modelId, {
                dtype: this.dtype,
            });

            // Pre-fetch available voices if possible
            if (this.tts && typeof this.tts.list_voices === 'function') {
                this.availableVoices = await this.tts.list_voices();
                this.logger.log(`Worker ${process.pid}: Found ${this.availableVoices.length} voices.`);
            } else {
                this.logger.warn(`Worker ${process.pid}: Could not retrieve voice list from model.`);
            }

            this.isModelReady = true;
            this.logger.log(`Worker ${process.pid}: TTS model loaded successfully.`);

        } catch (error) {
            this.logger.error(`Worker ${process.pid}: Failed to load TTS model: ${error.message}`, error.stack);
            this.isModelReady = false;
            // Optional: throw error to prevent service usage, or handle gracefully
            // throw new InternalServerErrorException('TTS model failed to load');
        }
    }

    isReady(): boolean {
        return this.isModelReady && this.tts !== null;
    }

    async generateTts(text: string, voice?: string): Promise<Buffer> {
        if (!this.isReady() || !this.tts) {
            this.logger.error(`Worker ${process.pid}: TTS service not ready, attempting to reload model...`);
            // Optionally attempt reload or throw a specific error
            await this.loadModel(); // Attempt reload
            if (!this.isReady() || !this.tts) {
                throw new InternalServerErrorException('TTS service is not available.');
            }
        }

        const selectedVoice = voice || this.defaultVoice;

        // Optional: Validate voice ID if list is available
        if (this.availableVoices.length > 0 && !this.availableVoices.some(v => v.id === selectedVoice)) {
            throw new BadRequestException(`Voice ID "${selectedVoice}" is not available.`);
        }


        this.logger.log(`Worker ${process.pid}: Generating speech for text: "${text.substring(0, 30)}..." using voice "${selectedVoice}"`);

        try {
            const startTime = Date.now();
            const audio: KokoroAudioOutput = await this.tts.generate(text, {
                voice: selectedVoice,
                // Add other generation options here if needed
            });
            const endTime = Date.now();
            this.logger.log(`Worker ${process.pid}: TTS generation took ${endTime - startTime}ms`);

            // --- Get Audio Buffer ---
            // This part depends heavily on the actual kokoro-js API
            if (typeof audio?.getWavBytes === 'function') {
                const buffer = await audio.getWavBytes();
                this.logger.log(`Worker ${process.pid}: Successfully generated WAV buffer (${(buffer.length / 1024).toFixed(2)} KB).`);
                return buffer;
            } else {
                // Fallback/Alternative: If only save exists
                /*
                const tempPath = `/tmp/audio-${process.pid}-${Date.now()}.wav`; // Use a unique path
                await audio.save(tempPath);
                const buffer = await fs.promises.readFile(tempPath);
                await fs.promises.unlink(tempPath); // Clean up temp file
                this.logger.log(`Worker ${process.pid}: Successfully generated WAV via temp file (${(buffer.length / 1024).toFixed(2)} KB).`);
                return buffer;
                */
                this.logger.error(`Worker ${process.pid}: Could not get audio buffer. The 'getWavBytes' method might be missing or named differently.`);
                throw new InternalServerErrorException('Failed to retrieve audio data from TTS result.');
            }

        } catch (error) {
            this.logger.error(`Worker ${process.pid}: Error during TTS generation: ${error.message}`, error.stack);
            if (error instanceof BadRequestException) { // Propagate validation errors
                throw error;
            }
            throw new InternalServerErrorException('Failed to generate speech.');
        }
    }

    getAvailableVoices(): KokoroVoice[] {
        if (!this.isReady()) {
            throw new InternalServerErrorException('TTS service is not available.');
        }
        return this.availableVoices;
    }
}