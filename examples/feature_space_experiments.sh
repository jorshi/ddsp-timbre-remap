ITERS=200

python feature_matching.py --data_path audio/carson_gant_drums/performance.wav --iters $ITERS
python feature_matching.py --data_path audio/drumloop_1.wav --iters $ITERS
python feature_matching.py --data_path audio/percussion_1.wav --iters $ITERS

python feature_matching_mfcc.py --data_path audio/carson_gant_drums/performance.wav --iters $ITERS
python feature_matching_mfcc.py --data_path audio/drumloop_1.wav --iters $ITERS
python feature_matching_mfcc.py --data_path audio/percussion_1.wav --iters $ITERS
