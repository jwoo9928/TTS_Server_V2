// src/app.module.ts
import { Module } from '@nestjs/common';
import { TtsModule } from './tts/tts.module';
// import { ConfigModule } from '@nestjs/config'; // Uncomment if using config

@Module({
    imports: [
        // ConfigModule.forRoot({ isGlobal: true }), // Uncomment if using config
        TtsModule,
    ],
    controllers: [],
    providers: [],
})
export class AppModule { }