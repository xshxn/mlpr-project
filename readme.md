MLPR Project Git repo

-Eshan, Samyaka, Kavish

Code to create manifest is in the Utils directory
All the preprocessing methods that we have tried are in the Utils directory.

Extract and save vectors used wav2vec2 to convert audio to vectors
Extract transcripts normalizes the transcripts and add it to the csv file instead of just a path to the transcript

Libri pretrain contains the code ran on kaggle to pre train our model on approximately 45-50 hours of libri speech data.
Torgo finetune contains the code ran on kaggle that fine tuned our model on the torgo dataset.

All the performance metrics are printed on the last cell of both the torgo finetune file and libri pretrain file.