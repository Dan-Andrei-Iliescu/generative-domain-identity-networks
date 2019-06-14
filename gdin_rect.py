import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pickle
import os
from tqdm import tqdm
tfd = tf.distributions

train_path = os.path.join('data', 'rect_rgb', 'rect_train.txt')
test_path = os.path.join('data', 'rect_rgb', 'rect_test.txt')
save_path = os.path.join('results', 'rect_final')

train_set = pickle.load(open(train_path, 'rb'))
test_set = pickle.load(open(test_path, 'rb'))

TRAIN_SIZE = train_set.shape[0]
TEST_SIZE = test_set.shape[0]
NUM_SAMPLES = train_set.shape[1]
SHAPE = train_set.shape[2]
CHANNELS = 3

NUM_FEATURES = SHAPE*SHAPE*CHANNELS
ID_SIZE = 4
DM_SIZE = 8
DEPTH_BIG = 256
DEPTH_SMALL = 64

LR = 0.0001
B1 = 0.5
EPOCHS = 10

train_set = train_set.reshape(TRAIN_SIZE, NUM_SAMPLES, NUM_FEATURES)
test_set = train_set.reshape(TEST_SIZE, NUM_SAMPLES, NUM_FEATURES)

TRAIN_MB = 128
SIGMA = 0.1


def plot(samples, title):
    height = samples.shape[0]*SHAPE
    width = samples.shape[1]*SHAPE
    grid = np.zeros((height, width, CHANNELS))

    for row in range(samples.shape[0]):
        for col in range(samples.shape[1]):
            grid[SHAPE*row:SHAPE*(row+1), SHAPE*col:SHAPE*(col+1)] = \
                samples[row, col].reshape(SHAPE, SHAPE, CHANNELS)

    path = os.path.join(save_path, title)
    plt.imsave(fname=path, arr=grid, cmap='gray')
    plt.close()


# Sample minibatch
def sample_idx(size, mb_size):
    permuted = np.random.permutation(size)
    idx = permuted[:mb_size]
    return idx


# Initialize wights
def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# Domain representer weights
rdm_w1 = tf.Variable(xavier_init([NUM_FEATURES, DEPTH_BIG]))
rdm_b1 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_BIG]))

rdm_w2 = tf.Variable(xavier_init([DEPTH_BIG, DEPTH_SMALL]))
rdm_b2 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_SMALL]))

rdm_w3 = tf.Variable(xavier_init([DEPTH_SMALL, DM_SIZE]))
rdm_b3 = tf.Variable(tf.zeros(shape=[1, 1, DM_SIZE]))

rdm_weights = [rdm_w1, rdm_b1, rdm_w2, rdm_b2, rdm_w3, rdm_b3]

# Identity representer weights
rid_w1 = tf.Variable(xavier_init([NUM_FEATURES, DEPTH_BIG]))
rid_b1 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_BIG]))

rid_w2 = tf.Variable(xavier_init([DEPTH_BIG, DEPTH_SMALL]))
rid_b2 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_SMALL]))

rid_w3 = tf.Variable(xavier_init([DEPTH_SMALL, ID_SIZE]))
rid_b3 = tf.Variable(tf.zeros(shape=[1, 1, ID_SIZE]))

rid_weights = [rid_w1, rid_b1, rid_w2, rid_b2, rid_w3, rid_b3]

# Encoder weights
e_w1 = tf.Variable(xavier_init([DM_SIZE+1, DEPTH_SMALL]))
e_b1 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_SMALL]))

e_w2 = tf.Variable(xavier_init([DEPTH_SMALL, DM_SIZE]))
e_b2 = tf.Variable(tf.zeros(shape=[1, 1, DM_SIZE]))

e_w3 = tf.Variable(xavier_init([DEPTH_SMALL, DM_SIZE]))
e_b3 = tf.Variable(tf.zeros(shape=[1, 1, DM_SIZE]))

e_weights = [e_w1, e_b1, e_w2, e_b2, e_w2, e_w3]

# Factor weights
f_w1 = tf.Variable(xavier_init([DM_SIZE+NUM_FEATURES, DEPTH_BIG]))
f_b1 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_BIG]))

f_w2 = tf.Variable(xavier_init([DEPTH_BIG, DEPTH_SMALL]))
f_b2 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_SMALL]))

f_w3 = tf.Variable(xavier_init([DEPTH_SMALL, ID_SIZE]))
f_b3 = tf.Variable(tf.zeros(shape=[1, 1, ID_SIZE]))

f_w4 = tf.Variable(xavier_init([DEPTH_SMALL, ID_SIZE]))
f_b4 = tf.Variable(tf.zeros(shape=[1, 1, ID_SIZE]))

f_weights = [f_w1, f_b1, f_w2, f_b2, f_w3, f_b3, f_w4, f_b4]

# Adversarial encoder weights
a_w1 = tf.Variable(xavier_init([ID_SIZE+1, DEPTH_SMALL]))
a_b1 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_SMALL]))

a_w2 = tf.Variable(xavier_init([DEPTH_SMALL, ID_SIZE]))
a_b2 = tf.Variable(tf.zeros(shape=[1, 1, ID_SIZE]))

a_w3 = tf.Variable(xavier_init([DEPTH_SMALL, ID_SIZE]))
a_b3 = tf.Variable(tf.zeros(shape=[1, 1, ID_SIZE]))

a_weights = [a_w1, a_b1, a_w2, a_b2, a_w3, a_b3]

# Generator weights
g_w1 = tf.Variable(xavier_init([DM_SIZE+ID_SIZE, DEPTH_SMALL]))
g_b1 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_SMALL]))

g_w2 = tf.Variable(xavier_init([DEPTH_SMALL, DEPTH_BIG]))
g_b2 = tf.Variable(tf.zeros(shape=[1, 1, DEPTH_BIG]))

g_w3 = tf.Variable(xavier_init([DEPTH_BIG, NUM_FEATURES]))
g_b3 = tf.Variable(tf.zeros(shape=[1, 1, NUM_FEATURES]))

g_weights = [g_w1, g_b1, g_w2, g_b2, g_w3, g_b3]


# Graph
def rep_dm(x):
    rdm_h1 = tf.linalg.einsum('ijk,kl->ijl', x, rdm_w1) + rdm_b1
    rdm_h1 = tf.nn.leaky_relu(rdm_h1)

    rdm_h2 = tf.linalg.einsum('ijk,kl->ijl', rdm_h1, rdm_w2) + rdm_b2
    rdm_h2 = tf.nn.leaky_relu(rdm_h2)

    rdm_h3 = tf.linalg.einsum('ijk,kl->ijl', rdm_h2, rdm_w3) + rdm_b3

    return rdm_h3


def rep_id(x):
    rid_h1 = tf.linalg.einsum('ijk,kl->ijl', x, rid_w1) + rid_b1
    rid_h1 = tf.nn.leaky_relu(rid_h1)

    rid_h2 = tf.linalg.einsum('ijk,kl->ijl', rid_h1, rid_w2) + rid_b2
    rid_h2 = tf.nn.leaky_relu(rid_h2)

    rid_h3 = tf.linalg.einsum('ijk,kl->ijl', rid_h2, rid_w3) + rid_b3

    return rid_h3


def encoder(rep, num_x):
    num_x = tf.cast(tf.broadcast_to(
        num_x, shape=[tf.shape(rep)[0], tf.shape(rep)[1], 1]
    ), dtype=tf.float32)
    e_h0 = tf.concat([rep, num_x], axis=2)

    e_h1 = tf.linalg.einsum('ijk,kl->ijl', e_h0, e_w1) + e_b1
    e_h1 = tf.nn.leaky_relu(e_h1)

    e_h2 = tf.linalg.einsum('ijk,kl->ijl', e_h1, e_w2) + e_b2
    e_h3 = tf.linalg.einsum('ijk,kl->ijl', e_h1, e_w3) + e_b3

    return e_h2, e_h3


def factor(dm, y):
    f_h0 = tf.concat([dm, y], axis=2)

    f_h1 = tf.linalg.einsum('ijk,kl->ijl', f_h0, f_w1) + f_b1
    f_h1 = tf.nn.leaky_relu(f_h1)

    f_h2 = tf.linalg.einsum('ijk,kl->ijl', f_h1, f_w2) + f_b2
    f_h2 = tf.nn.leaky_relu(f_h2)

    f_h3 = tf.linalg.einsum('ijk,kl->ijl', f_h2, f_w3) + f_b3
    f_h4 = tf.linalg.einsum('ijk,kl->ijl', f_h2, f_w4) + f_b4

    return f_h3, f_h4


def adversary(rep, num_x):
    num_x = tf.cast(tf.broadcast_to(
        num_x, shape=[tf.shape(rep)[0], tf.shape(rep)[1], 1]
    ), dtype=tf.float32)
    a_h0 = tf.concat([rep, num_x], axis=2)

    a_h1 = tf.linalg.einsum('ijk,kl->ijl', a_h0, a_w1) + a_b1
    a_h1 = tf.nn.leaky_relu(a_h1)

    a_h2 = tf.linalg.einsum('ijk,kl->ijl', a_h1, a_w2) + a_b2
    a_h3 = tf.linalg.einsum('ijk,kl->ijl', a_h1, a_w3) + a_b3

    return a_h2, a_h3


def generator(dm, id):
    g_h0 = tf.concat([dm, id], axis=2)

    g_h1 = tf.linalg.einsum('ijk,kl->ijl', g_h0, g_w1) + g_b1
    g_h1 = tf.nn.leaky_relu(g_h1)

    g_h2 = tf.linalg.einsum('ijk,kl->ijl', g_h1, g_w2) + g_b2
    g_h2 = tf.nn.leaky_relu(g_h2)

    g_h3 = tf.linalg.einsum('ijk,kl->ijl', g_h2, g_w3) + g_b3
    g_h3 = tf.sigmoid(g_h3)

    return g_h3


# Input placeholders
x = tf.placeholder(tf.float32, shape=[None, None, NUM_FEATURES])
y = tf.placeholder(tf.float32, shape=[None, None, NUM_FEATURES])

batch_size = tf.shape(x)[0]
num_x = tf.shape(x)[1]
num_y = tf.shape(y)[1]

# Prior sampling net
# Produce combined representation
x_dm_rep = rep_dm(x)
prior_rep = tf.reshape(
    tf.math.reduce_sum(x_dm_rep, axis=1),
    shape=(batch_size, 1, DM_SIZE)
)

# Domain prior
prior_dm_loc, prior_dm_scale = encoder(prior_rep, num_x)
prior_dm_scale = SIGMA * tf.ones_like(prior_dm_scale)
prior_dm_dist = tfd.Normal(loc=prior_dm_loc, scale=prior_dm_scale)
prior_dm = prior_dm_dist.sample()
prior_dm_broad = tf.broadcast_to(prior_dm, [batch_size, num_y, DM_SIZE])

# Identity prior
prior_id = tf.random.normal(
    shape=[batch_size, num_y, ID_SIZE], mean=0., stddev=1.
)
prior_id_loc = tf.zeros_like(prior_id)
prior_id_scale = SIGMA * tf.ones_like(prior_id)
prior_id_dist = tfd.Normal(loc=prior_id_loc, scale=prior_id_scale)

# Generate from prior latent
same = tf.random.normal(
    shape=[1, num_y, ID_SIZE], mean=0., stddev=1.
)
same = tf.broadcast_to(same, [batch_size, num_y, ID_SIZE])

same = tf.reshape(tf.range(start=-1., limit=1., delta=(
    2./tf.cast(num_y, dtype=tf.float32)
)), [1, num_y, 1])
same = tf.broadcast_to(same, [batch_size, num_y, ID_SIZE])
prior_gen = generator(prior_dm_broad, same)

# Inference sampling net
# Produce combined representation
y_dm_rep = rep_dm(y)
inf_rep = prior_rep + y_dm_rep

# Domain inference
inf_dm_loc, inf_dm_scale = encoder(inf_rep, num_x + 1)
inf_dm_scale = SIGMA * tf.ones_like(inf_dm_scale)
inf_dm_dist = tfd.Normal(loc=inf_dm_loc, scale=inf_dm_scale)
inf_dm = inf_dm_dist.sample()
inf_dm_broad = tf.broadcast_to(inf_dm, [batch_size, num_y, DM_SIZE])

# Identity inference
inf_id_loc, inf_id_scale = factor(inf_dm_broad, y)
inf_id_scale = SIGMA * tf.ones_like(inf_id_scale)
inf_id_dist = tfd.Normal(loc=inf_id_loc, scale=inf_id_scale)

# Identity filter
x_id_rep = rep_id(x)
adv_id_rep = tf.reshape(
    tf.math.reduce_sum(x_id_rep, axis=1),
    shape=(batch_size, 1, ID_SIZE)
)
adv_id_rep_broad = tf.broadcast_to(adv_id_rep, [batch_size, num_y, ID_SIZE])
adv_id_loc, adv_id_scale = adversary(adv_id_rep_broad, num_x)
adv_id_scale = SIGMA * tf.ones_like(adv_id_scale)
adv_id_dist = tfd.Normal(loc=adv_id_loc, scale=adv_id_scale)

# Filtered identity
fil_id_loc = inf_id_loc - adv_id_loc
fil_id_scale = inf_id_scale / adv_id_scale
fil_id_scale = SIGMA * tf.ones_like(fil_id_scale)
fil_id_dist = tfd.Normal(loc=fil_id_loc, scale=fil_id_scale)
fil_id = fil_id_dist.sample()

# Generate from inferred latent
inf_gen = generator(inf_dm_broad, fil_id)


# Losses
def mse_loss(labels, preds):
    return tf.losses.mean_squared_error(labels=labels, predictions=preds)


def l1_loss(labels, preds):
    return tf.reduce_mean(tf.abs(labels - preds))


def ce_loss(labels, logits):
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
    )


# Loss
l_u = l1_loss(inf_dm_loc, prior_dm_loc)
l_v = mse_loss(inf_id_loc, prior_id_loc)
l_f = l1_loss(adv_id_loc, inf_id_loc)

# Reconstruction likelihood loss
rec_loss = l1_loss(y, inf_gen)

reg_dm_loss = l1_loss(prior_dm_loc, inf_dm_loc)
reg_id_loss = tf.reduce_mean(inf_id_loc**2)

# Optimisers
optimiser = tf.train.AdamOptimizer(learning_rate=LR, beta1=B1)
re_solver = optimiser.minimize(
    rec_loss, var_list=rdm_weights+e_weights
)
f_solver = optimiser.minimize(rec_loss + 0.01*l_v, var_list=f_weights)
g_solver = optimiser.minimize(rec_loss, var_list=g_weights)
ra_solver = optimiser.minimize(l_f, var_list=rid_weights+a_weights)

# Sessions
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Testing samples
test_mb = test_set[:TRAIN_MB]
plot(test_mb[:8], 'real.png')

for it in tqdm(range(100000)):
    # Get conditions and targets
    idx = sample_idx(TRAIN_SIZE, TRAIN_MB)
    train_mb = train_set[idx]
    ratio = np.random.uniform(0, 1)
    sel = np.random.binomial(1, ratio, NUM_SAMPLES)
    X = train_mb[:, np.where(sel)].reshape(TRAIN_MB, -1, NUM_FEATURES)
    Y = train_mb[:, np.where(1 - sel)].reshape(TRAIN_MB, -1, NUM_FEATURES)

    sess.run(
        [re_solver, f_solver, g_solver, ra_solver],
        feed_dict={x: X, y: Y}
    )

    train_rec, train_id, train_dm = sess.run(
        [rec_loss, reg_id_loss, reg_dm_loss],
        feed_dict={x: X, y: Y}
    )

    if it % 1000 == 999:
        print(' ')
        print('Iteration {}'.format(it))
        print('Train reconstruction loss: {}'.format(train_rec))
        print('Train domain loss: {}'.format(train_dm))
        print('Train id loss: {}'.format(train_id))
        X = test_mb[:, :int(NUM_SAMPLES/2)]
        Y = test_mb[:, int(NUM_SAMPLES/2):]
        test_prior, test_inf = sess.run(
            [prior_gen, inf_gen], feed_dict={x: X, y: Y}
        )
        domain_error = np.mean(
            np.abs(np.mean(test_prior, axis=1) - np.mean(test_inf, axis=1))
        )
        print(' ')
        print('Domain error: {}'.format(domain_error))
        test_sample = np.concatenate((test_prior, test_inf), axis=1)
        plot(test_sample[:8], '{}.png'.format(it))
        plt.close()
