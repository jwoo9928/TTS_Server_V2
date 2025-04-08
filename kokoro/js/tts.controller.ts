// src/tts/tts.controller.ts
import { Controller, Post, Body, Res, Get, HttpCode, HttpStatus, ServiceUnavailableException, Header } from '@nestjs/common';
import { TtsService } from './tts.service';
import { CreateTtsDto } from './dto/create-tts.dto';
import { Response } from 'express';
import { ApiTags, ApiOperation, ApiResponse, ApiBody, ApiProduces } from '@nestjs/swagger';

@ApiTags('tts') // Group endpoints under 'tts' tag in Swagger
@Controller('tts')
export class TtsController {
    constructor(private readonly ttsService: TtsService) { }

    @Post()
    @HttpCode(HttpStatus.OK) // Set default success code to 200
    @Header('Content-Type', 'audio/wav') // Set the correct content type
    @ApiOperation({ summary: 'Generate speech from text' })
    @ApiBody({ type: CreateTtsDto })
    @ApiProduces('audio/wav') // Indicate the response MIME type
    @ApiResponse({ status: 200, description: 'Successfully generated WAV audio.' })
    @ApiResponse({ status: 400, description: 'Bad Request - Invalid input data (e.g., missing text, invalid voice).' })
    @ApiResponse({ status: 500, description: 'Internal Server Error - Failed to generate speech.' })
    @ApiResponse({ status: 503, description: 'Service Unavailable - TTS model not ready.' })
    async generateSpeech(
        @Body() createTtsDto: CreateTtsDto,
        @Res({ passthrough: true }) res: Response, // Use passthrough for easier buffer return
    ): Promise<Buffer> { // Return Buffer directly
        if (!this.ttsService.isReady()) {
            throw new ServiceUnavailableException('TTS service is initializing or failed to load. Please try again later.');
        }
        try {
            const audioBuffer = await this.ttsService.generateTts(createTtsDto.text, createTtsDto.voice);
            // Set content disposition if you want the browser to suggest a filename
            // res.setHeader('Content-Disposition', 'attachment; filename="speech.wav"');
            return audioBuffer; // NestJS handles sending the buffer with the correct headers
        } catch (error) {
            // Let NestJS default exception filters handle InternalServerErrorException and BadRequestException
            // Or re-throw specific HTTP exceptions if needed
            throw error;
        }
    }

    @Get('voices')
    @ApiOperation({ summary: 'List available TTS voices' })
    @ApiResponse({ status: 200, description: 'List of available voices.', type: [Object] }) // Adjust type based on KokoroVoice structure
    @ApiResponse({ status: 503, description: 'Service Unavailable - TTS model not ready.' })
    getVoices() {
        if (!this.ttsService.isReady()) {
            throw new ServiceUnavailableException('TTS service is initializing or failed to load.');
        }
        try {
            return this.ttsService.getAvailableVoices();
        } catch (error) {
            throw error; // Propagate potential errors from service
        }
    }

    @Get('health')
    @ApiOperation({ summary: 'Check if the TTS service is ready' })
    @ApiResponse({ status: 200, description: 'Service is ready.' })
    @ApiResponse({ status: 503, description: 'Service is not ready.' })
    checkHealth() {
        if (this.ttsService.isReady()) {
            return { status: 'ok', message: 'TTS service is ready.' };
        } else {
            throw new ServiceUnavailableException('TTS service is not ready.');
        }
    }
}