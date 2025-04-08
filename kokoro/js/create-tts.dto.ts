// src/tts/dto/create-tts.dto.ts
import { ApiProperty } from '@nestjs/swagger';
import { IsString, IsNotEmpty, IsOptional } from 'class-validator';

export class CreateTtsDto {
    @ApiProperty({
        description: 'The text to synthesize.',
        example: 'Life is like a box of chocolates.',
    })
    @IsString()
    @IsNotEmpty()
    text: string;

    @ApiProperty({
        description: 'The voice ID to use for synthesis. Uses default if not provided.',
        example: 'af_bella',
        required: false,
    })
    @IsString()
    @IsOptional()
    voice?: string; // Make voice optional

    // Add other potential options like speed, pitch etc. as needed
}