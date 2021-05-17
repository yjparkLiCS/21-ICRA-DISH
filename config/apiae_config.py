
APIAE_PARAMS = dict(
    n_x=90,  # dimension of x; observation
    n_z=3,  # dimension of z; latent space
    n_u=1,  # dimension of u; control

    K=10,  # the number of time steps
    R=1,  # the number of adaptations
    L=32,  # the number of trajectory sampled

    dt=.1,  # time interval
    ur=.1,  # update rate
    lr=1e-3,  # learning rate
    
    u_ratio=2.0,
    taus=0.1,
)

TRAINING_EPOCHS = 5000
OFFSET_STD = 1e-5
