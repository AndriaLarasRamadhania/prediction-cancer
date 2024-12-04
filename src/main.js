const tf = require('@tensorflow/tfjs-node');
require('dotenv').config();
const Hapi = require('@hapi/hapi');
const crypto = require('crypto');
const { Firestore } = require('@google-cloud/firestore');
// const { ClientError, InputError } = require('./errorHandling');
const predictClassification = require('./inferenceService');

// Load model from environment-specified URL
async function loadModel() {
    try {
        return await tf.loadGraphModel(process.env.MODEL_URL);
    } catch (error) {
        console.error('Error loading TensorFlow model:', error);
        throw new Error('Model loading failed');
    }
}

// Store prediction data in Firestore
async function storeData(id, data) {
    const db = new Firestore();
    const predictCollection = db.collection('prediction');
    try {
        await predictCollection.doc(id).set(data);
    } catch (error) {
        console.error('Error storing data in Firestore:', error);
        throw new Error('Data storage failed');
    }
}

// Handle prediction requests
async function postPredictHandler(request, h) {
    const { image } = request.payload;
    const { model } = request.server.app;

    console.log('Received payload size:', image.length);

    if (image.length > 1000000) {
        return h.response({
            status: 'fail',
            message: 'Payload content length exceeds maximum allowed: 1000000',
        }).code(413); // Payload Too Large
    }

    try {
        const { label, suggestion } = await predictClassification(model, image);
        const id = crypto.randomUUID();
        const createdAt = new Date().toISOString();

        const data = { id, result: label, suggestion, createdAt };
        await storeData(id, data);

        return h.response({
            status: 'success',
            message: 'Prediction successful',
            data,
        }).code(201); // Created
    } catch (error) {
        console.error('Prediction error:', error);
        return h.response({
            status: 'fail',
            message: 'Error occurred during prediction',
        }).code(500); // Internal Server Error
    }
}

(async () => {
    const server = Hapi.server({
        port: 3000,
        host: '0.0.0.0',
        routes: {
            cors: { origin: ['*'] },
        },
    });


    try {
        const model = await loadModel();
        server.app.model = model;

        server.route([
            {
                path: '/predict',
                method: 'POST',
                handler: postPredictHandler,
                options: {
                    payload: {
                        allow: 'multipart/form-data',
                        multipart: true,
                        maxBytes: 1000000,
                        parse: true,
                    },
                },
            },
        ]);

        // Unified error handling
        server.ext('onPreResponse', (request, h) => {
            const response = request.response;

            // Handle custom errors
            if (response instanceof ClientError) {
                return h.response({
                    status: 'fail',
                    message: response.message,
                }).code(response.statusCode);
            }

            // Handle internal errors (Boom errors)
            if (response.isBoom) {
                console.error('Server error:', response.message);
                return h.response({
                    status: 'fail',
                    message: response.message,
                }).code(response.output.statusCode);
            }

            return h.continue;
        });

        await server.start();
        console.log(`Server started at: ${server.info.uri}`);
    } catch (error) {
        console.error('Error starting server:', error);
        process.exit(1);
    }
})();
