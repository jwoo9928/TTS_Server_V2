// src/main.ts
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { SwaggerModule, DocumentBuilder } from '@nestjs/swagger';
import { ValidationPipe, Logger } from '@nestjs/common';
import * as cluster from 'cluster'; // Use default import
import * as os from 'os';
// eslint-disable-next-line @typescript-eslint/no-var-requires
const clusterModule = require('cluster') as unknown as cluster.Cluster; // Handle commonjs/esm interop

const numCPUs = os.cpus().length; // Get number of CPU cores

async function bootstrap() {
    const app = await NestFactory.create(AppModule);
    const logger = new Logger('Bootstrap');

    // Global Validation Pipe
    app.useGlobalPipes(new ValidationPipe({
        whitelist: true, // Strip properties not in DTO
        transform: true, // Automatically transform payloads to DTO instances
    }));

    // Swagger Setup (only in worker processes)
    const config = new DocumentBuilder()
        .setTitle('Kokoro TTS API')
        .setDescription('API for generating speech using kokoro-js with parallel processing.')
        .setVersion('1.0')
        .addTag('tts')
        .build();
    const document = SwaggerModule.createDocument(app, config);
    SwaggerModule.setup('api-docs', app, document); // Setup Swagger UI at /api-docs

    const port = process.env.PORT || 3000;
    await app.listen(port);
    logger.log(`Worker ${process.pid} started and listening on port ${port}`);
    logger.log(`Swagger UI available at http://localhost:${port}/api-docs`);
}

// --- Clustering Logic ---
if (clusterModule.isPrimary) {
    const logger = new Logger('ClusterPrimary');
    logger.log(`Primary process ${process.pid} is running`);
    logger.log(`Forking ${numCPUs} worker processes...`);

    // Fork workers for each CPU core
    for (let i = 0; i < numCPUs; i++) {
        clusterModule.fork();
    }

    clusterModule.on('exit', (worker, code, signal) => {
        logger.warn(`Worker ${worker.process.pid} died with code ${code} and signal ${signal}. Forking a new worker...`);
        clusterModule.fork(); // Replace the dead worker
    });

} else {
    // Worker processes run the NestJS application
    bootstrap().catch(err => {
        const logger = new Logger('BootstrapWorkerError');
        logger.error(`Error starting worker ${process.pid}: ${err.message}`, err.stack);
        process.exit(1); // Exit if bootstrap fails
    });
}