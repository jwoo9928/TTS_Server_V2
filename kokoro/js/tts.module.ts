// src/tts/tts.module.ts
import { Module } from '@nestjs/common';
import { TtsService } from './tts.service';
import { TtsController } from './tts.controller';

@Module({
    providers: [TtsService],
    controllers: [TtsController]
})
export class TtsModule { }