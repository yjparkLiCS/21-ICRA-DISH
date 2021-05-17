
import os
import argparse
import logging

import numpy as np

from apiae.apiae_net import APIAE
from config import DATA_DIR, APIAE_PARAMS, TRAINING_EPOCHS, OFFSET_STD

logging.basicConfig(level=logging.DEBUG)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rollout_fname', type=str,
        default='rollout0.npz'
    )
    parser.add_argument(
        '--weight_fname', type=str,
        default='apiae0.pkl'
    )
    args = parser.parse_args()
    return vars(args)


def main():
    kwargs = parse_args()
    train(**kwargs)


def train(rollout_fname, weight_fname):
    # load data
    rollout_data = np.load(os.path.join(DATA_DIR['rollout'], rollout_fname))
    uffseq_in = rollout_data['Hall'][:, :, :-1, :]
    xref_orig = rollout_data['Dall'].reshape((-1, APIAE_PARAMS['n_x']))  # (BT, D)
    xref_mean = np.mean(xref_orig, axis=0, keepdims=True)  # (1, D)
    xref_std = np.std(xref_orig, axis=0, keepdims=True) + OFFSET_STD  # (1, D)
    xref_normalized = (xref_orig - xref_mean) / xref_std
    xref = xref_normalized.reshape(-1, 1, APIAE_PARAMS['K'], APIAE_PARAMS['n_x'])
    logging.info('Rollout is loaded.')

    # save scale params
    scale_fname = os.path.join(
        DATA_DIR['scale'], 'scale_' + rollout_fname
    )
    np.savez(
        scale_fname,
        x_mean=xref_mean.reshape((1, 1, 1, -1)),
        x_std=xref_std.reshape((1, 1, 1, -1))
    )
    logging.info("Scale is saved.")

    # build apiae graph
    apiae_net = APIAE(scale_fname=scale_fname, **APIAE_PARAMS)
    logging.info('APIAE graph is built.')

    # initialize dynamics network for the better stability
    z_initials = [10 * np.random.normal(loc=0.0, scale=1.0, size=(5000, 1)) for _ in range(3)]
    zref_init = np.concatenate(z_initials, axis=1)
    a_init = -1.0 * np.eye(3)
    b_init = np.zeros((3, 1))
    sigma_init = .01 * np.ones((APIAE_PARAMS['n_z'], APIAE_PARAMS['n_u']))
    apiae_net.dynNet.initialize(
        apiae_net.sess, zref_init, a_init, b_init, sigma_init,
        batch_size=500, training_epochs=500, display_step=100
    )
    logging.info('Dynamic network is initialized.')

    # train model
    n_rollout = xref.shape[0]
    batch_size = np.minimum(512, n_rollout)
    total_batch = int(n_rollout / batch_size)
    for epoch in range(1, TRAINING_EPOCHS + 1):
        total_loss = 0.
        nperm = np.random.permutation(n_rollout)
        for i in range(total_batch):
            minibatch_idx = nperm[i * batch_size:(i + 1) * batch_size]
            batch_xs = xref[minibatch_idx, :, :, :]
            batch_u = uffseq_in[minibatch_idx, :, :, :]
            loss, _, _, _ = apiae_net.partial_fit(batch_xs, batch_u)
            total_loss += loss

        if epoch % 100 == 0:
            # log
            logging.info('Epoch={:04d}, Bound={:.2f}'.format(epoch, -total_loss))

        if epoch % 1000 == 0:
            loss, museq, uffseq, xseq = apiae_net.partial_fit(
                batch_xs, batch_u, return_value=True
            )

            # save params
            apiae_net.save_weights(os.path.join(DATA_DIR['weight'], weight_fname))
            logging.info('APIAE weights are saved.')

            # save latent samples
            latent_fname = os.path.join(
                DATA_DIR['latent'], 'latent_' + rollout_fname
            )
            np.savez(
                latent_fname,
                x_true=np.reshape(batch_xs, (-1, APIAE_PARAMS['n_x'])) * xref_std + xref_mean,
                x_infer=np.reshape(xseq, (-1, APIAE_PARAMS['n_x'])) * xref_std + xref_mean,
                z_infer=np.reshape(museq, (-1, APIAE_PARAMS['n_z'])),
                u_infer=np.reshape(uffseq, (-1, APIAE_PARAMS['n_u'])),
                u_true=np.reshape(batch_u, (-1, APIAE_PARAMS['n_u']))
            )
            logging.info('Latent samples are saved.')


if __name__ == '__main__':
    main()
