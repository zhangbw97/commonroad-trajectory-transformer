N_BINS = 100
EMBD_DIM=12 # How to choose? Even 2^32 is too large for representing TRANSITION_DIM*SEQUENCE_LENGTH*N_BINS
TRANSITION_DIM=8 # traffic prediction
# OBSERVATION_DIM= 6
OBSERVATION_DIM= 6 # traffic prediction
# OCC_RELATED_OBSERVATION_DIM = 2
OCC_EMBEDDING_DIM = 32
ACTION_DIM=2
SEQUENCE_LENGTH=40

