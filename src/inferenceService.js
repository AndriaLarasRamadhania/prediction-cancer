const tf = require('@tensorflow/tfjs-node');
//const InputError = require('./inputError'); // Impor InputError
// const { ClientError, InputError } = require('./errorHandling');

async function predictClassification(model, image) {
    try {
        const tensor = tf.node
            .decodeJpeg(image)
            .resizeNearestNeighbor([224, 224])
            .expandDims()
            .toFloat();

        const prediction = model.predict(tensor);
        const score = (await prediction.data())[0];
        const label = score > 0.5 ? 'Cancer' : 'Non-cancer';

        const suggestion = label === 'Cancer'
            ? "Segera periksa ke dokter!"
            : "Penyakit kanker tidak terdeteksi.";

        return { label, suggestion };
    } catch (error) {
        console.error("Prediction error: ", error.message);
        throw new ClientError(`Terjadi kesalahan input: ${error.message}`);
    }
}

module.exports = predictClassification;
