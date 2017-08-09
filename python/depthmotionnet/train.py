import tensorflow as tf
import numpy as np


def l1(prediction, gt):
    """
    Computes the L1 norm, normalized wrt the number of elements (BS * H * W * C)
    """
    # assert all finite
    assert_gt_op = tf.Assert(tf.reduce_all(tf.is_finite(gt)), [gt])
    assert_pred_op = tf.Assert(tf.reduce_all(tf.is_finite(prediction)), [prediction])

    with tf.control_dependencies([assert_gt_op, assert_pred_op]):
        diff = prediction - gt
        num_pixels = tf.size(diff, out_type=tf.float32)
        return tf.reduce_sum(tf.abs(diff)) / num_pixels


def l2(prediction, gt, normalize=True):
    """
    Computes the L2 norm, normalized wrt the number of elements (BS * H * W * C)
    """
    # assert all finite
    assert_gt_op = tf.Assert(tf.reduce_all(tf.is_finite(gt)), [gt])
    assert_pred_op = tf.Assert(tf.reduce_all(tf.is_finite(prediction)), [prediction])

    with tf.control_dependencies([assert_gt_op, assert_pred_op]):
        diff = prediction - gt
        if normalize:
            num_pixels = tf.size(diff, out_type=tf.float32)
            return tf.sqrt(tf.reduce_sum(tf.square(diff)) / num_pixels)
        else:
            return tf.sqrt(tf.reduce_sum(tf.square(diff)))


def scale_invariant_gradient_loss(prediction, gt):
    def discrete_scale_invariant_gradient(f, h):
        """
        Calculates the discrete scale invariant gradient of f with spacing h
        """
        _, height, width, _ = f.shape.as_list()

        # Pad the input width and height to allow for the spacing
        padded_f = tf.pad(f, [[0, 0], [0, h], [0, h], [0, 0]])

        # f(i + h, j)
        f_ih_j = padded_f[:, 0:height, h:width + h, :]

        # (f(i + h, j) - f(i, j)) / (|f(i + h, j)| + |f(i, j)|)
        i = (f_ih_j - f) / (tf.abs(f_ih_j) + tf.abs(f))

        # f(i, j + h)
        f_i_jh = padded_f[:, h:height + h, 0:width, :]

        # (f(i, j + h) - f(i, j)) / (|f(i, j + h)| + |f(i, j)|)
        j = (f_i_jh - f) / (tf.abs(f_i_jh) + tf.abs(f))

        return tf.stack([i, j])

    all_losses = []
    hs = [1, 2, 4, 8, 16]
    for h in hs:
        pred_grad = discrete_scale_invariant_gradient(prediction)
        gt_grad = discrete_scale_invariant_gradient(gt)
        all_losses.append(l2(pred_grad, gt_grad_i, normalize=False))
    return tf.reduce_sum(tf.accumulate_n(all_losses))


def loss(predictions, gt):
    """
    Returns the total loss for the DeMoN architecture

    predictions: dictionary with 'depth', 'normals', 'flow', 'flow_conf', 'rotation', 'translation'
    gt: dictionary with 'depth', 'normals', 'flow', 'rotation', and 'translation'
    """
    depth_loss = l1(predictions['depth'], gt['depth'])
    normal_loss = l2(predictions['normals'], gt['normals'])
    flow_loss = l2(predictions['flow'], gt['flow'])

    # Calculate gt flow confidence and loss from flow prediction
    gt_flow_conf = tf.exp(-tf.abs(predictions['flow'] - gt['flow']))
    flow_conf_loss = l1(predictions['flow_conf'], gt_flow_conf)

    rotation_loss = l2(predictions['rotation'], gt['rotation'])
    translation_loss = l2(predictions['translation'], gt['translation'])

    # Calculate scale invariant gradient losses
    flow_si_loss = scale_invariant_gradient_loss(predictions['flow'], gt['flow'])
    depth_si_loss = scale_invariant_gradient_loss(predictions['inv_depth'], gt['inv_depth'])

    # Add the weighted loss to tf internal list of losses
    losses = [inv_depth_loss, normal_loss, flow_loss,
              flow_conf_loss, rotation_loss, translation_loss, flow_si_loss, depth_si_loss]
    loss = tf.losses.compute_weighted_loss(losses, [300, 100, 1000, 1000, 160, 15, 1000, 1500])

    # Get all losses (weighted sum + regularization)
    return tf.losses.get_total_loss()


def finetune(session, image1, image2, depth, normals, flow, rotation, translation, learning_rate, max_iterations, weights_dir, log_dir):
    """
    Finetune a DeMoN architecture by updating the weights of the iterative network.

    Performs forward pass of bootstrap network with minibatches of size 8.

    We simulate 4 iterations of the iterative network by appending predictions from previous iterations to the minibatch.

    Parameters:
        session: a TF session instance
        image1: image1 as tf variable with dimensions 192x256 as (BS, H, W, K) scaled between -0.5, 0.5
        image2: image2 as tf variable with dimensions 192x256 as (BS, H, W, K) scaled between -0.5, 0.5
        depth: ground truth depth as (BS, 192, 256, 1)
        normals: ground truth normals as (BS, 192, 256, 3)
        flow: ground truth optical flow as (BS, 192, 256, 2)
        rotation: ground truth rotation as (BS, 3)
        translation: ground truth translation as (BS, 3)
        learning_rate: learning rate for optimizer
        max_iterations: maximum iterations to train for
        weights_dir: an absolute path to the directory containing 'demon_original' weights
        log_dir: an absolute path to a directory to store loss information for TB and checkpoints
    """

    minibatch_size = 8

    # Assume we're training on a GPU, so our data should be (BS, K, H, W)
    data_format = 'channels_first'

    # Holds the inputs to the iterative net. We simulate 4 iterations of the iterative network
    # by simply appending results from iteration i to the inputs of iteration i + 1 for i = 1...4
    iterative_image_pair_ = None
    iterative_image2_2_ = None
    iterative_depth2_ = None
    iterative_normal2_ = None
    iterative_rotation_ = None
    iterative_translation_ = None
    # gt numpy arrays
    iterative_depth2_gt = None
    iterative_normal2_gt = None
    iterative_flow2_gt = None
    iterative_rotation_gt = None
    iterative_translation_gt = None

    # Tensorflow gt placeholders for loss fn
    placeholder_depth2_gt = tf.placeholder(tf.float32, shape=(None, 192, 256, 1))
    placeholder_normal2_gt = tf.placeholder(tf.float32, shape=(None, 192, 256, 3))
    placeholder_flow2_gt = tf.placeholder(tf.float32, shape=(None, 192, 256, 2))
    placeholder_rotation_gt = tf.placeholder(tf.float32, shape=(None, 3))
    placeholder_translation_gt = tf.placeholder(tf.float32, shape=(None, 3))

    # Init variables
    session.run(tf.global_variables_initializer())

    # Load weights
    saver = tf.train.Saver()
    saver.restore(session, os.path.join(weights_dir, 'demon_original'))

    # Prepare inputs
    image_pair = tf.concat([image1, image2], 3)
    image2_2 = tf.image.resize_images(image2, [48, 64])

    # Rearrange from (BS, H, W, K) to (BS, K, H, W)
    image_pair = tf.transpose(image_pair, perm=[0, 3, 1, 2])
    image2_2 = tf.transpose(image2_2, perm=[0, 3, 1, 2])

    # Resize ground truths and rearrange to (BS, K, H, W)
    depth_resize = tf.transpose(tf.image.resize_images(depth, [48, 64]), perm=[0, 3, 1, 2])
    normals_resize = tf.transpose(tf.image.resize_images(normals, [48, 64]), perm=[0, 3, 1, 2])
    flow_resize = tf.transpose(tf.image.resize_images(flow, [48, 64]), perm=[0, 3, 1, 2])

    # Construct bootstrap and iterative networks
    bootstrap_net = BootstrapNet(session, data_format)
    iterative_net = IterativeNet(session, data_format)

    # Forward pass of the iterative net (returning Tensors)
    iter_results = iterative_net.forward()

    # Predictions from iterative net
    iter_depth2 = iter_results['predict_depth2']
    iter_flow2 = iter_results['predict_flow2']
    iter_normal2 = iter_results['predict_normal2']
    iter_rotation = iter_results['predict_rotation']
    iter_translation = iter_results['predict_translation']

    # Setup input to the loss function
    predictions = {
        'depth': iter_depth2,
        'normals': iter_normal2,
        # the first two channels are flow...
        'flow': iter_flow2[:, 0:2, :, :],
        # ...and the last two channels are confidence
        'flow_conf': iter_flow2[:, 2:4, :, :],
        'rotation': iter_rotation,
        'translation': iter_translation,
    }
    gt = {
        'depth': placeholder_depth2_gt,
        'normals': placeholder_normal2_gt,
        'flow': placeholder_flow2_gt,
        'rotation': placeholder_rotation_gt,
        'translation': iterative_translation_g_tft
    }

    # Define our loss and optimizer
    total_loss = loss(predictions, gt)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_loss)

    # Create TB loss plot
    total_loss_summary = tf.summary.scalar("loss", total_loss)

    # Create saver to save new weights
    saver = tf.train.Saver()

    # Create writers to save loss information
    writer_t = tf.summary.FileWriter(log_dir, None)

    # Iterate through data
    for step in range(max_iterations):
        # Get our input data and gt data
        image_pair_, image2_2_ = sess.run([image_pair, image2_2])
        depth2_gt, normal2_gt, flow2_gt, rotation_gt, translation_gt = sess.run(
            [depth, normals, flow, rotation, translation])

        # Forward pass of bootstrap net (returning np arrays)
        bootstrap_result = bootstrap_net.eval(image_pair_, image2_2_)

        # Predictions from bootstrap net
        depth2 = bootstrap_result['predict_depth2']
        normal2 = bootstrap_result['predict_normal2']
        rotation = bootstrap_result['predict_rotation']
        translation = bootstrap_result['predict_translation']

        # Concatenate bootstrap net predictions to np array of iterative net inputs
        if iterative_image_pair_ is None:
            iterative_image_pair_ = image_pair_
            iterative_image2_2_ = image2_2_
            iterative_depth2_ = depth2
            iterative_normal2_ = normal2
            iterative_rotation_ = rotation
            iterative_translation_ = translation
            iterative_depth2_gt = depth2_gt
            iterative_normal2_gt = normal2_gt
            iterative_flow2_gt = flow2_gt
            iterative_rotation_gt = rotation_gt
            iterative_translation_gt = translation_gt
        else:
            iterative_image_pair_ = np.concatenate((image_pair_, iterative_image_pair_), axis=0)
            iterative_image2_2_ = np.concatenate((image2_2_, iterative_image2_2_), axis=0)
            iterative_depth2_ = np.concatenate((depth2, iterative_depth2_), axis=0)
            iterative_normal2_ = np.concatenate((normal2, iterative_normal2_), axis=0)
            iterative_rotation_ = np.concatenate((rotation, iterative_rotation_), axis=0)
            iterative_translation_ = np.concatenate((translation, iterative_translation_), axis=0)
            iterative_depth2_gt = np.concatenate((depth2_gt, iterative_depth2_gt), axis=0)
            iterative_normal2_gt = np.concatenate((normal2_gt, iterative_normal2_gt), axis=0)
            iterative_flow2_gt = np.concatenate((flow2_gt, iterative_flow2_gt), axis=0)
            iterative_rotation_gt = np.concatenate((rotation_gt, iterative_rotation_gt), axis=0)
            iterative_translation_gt = np.concatenate(
                (translation_gt, iterative_translation_gt), axis=0)

        # Create feed dictionary
        feed_dict = {
            'placeholder_image_pair': iterative_image_pair_,
            'placeholder_image2_2': iterative_image2_2_,
            'placeholder_depth2': iterative_depth2_,
            'placeholder_normal2': iterative_normal2_,
            'placeholder_rotation': iterative_rotation_,
            'placeholder_translation': iterative_translation_,
            placeholder_depth2_gt: iterative_depth2_gt,
            placeholder_normal2_gt: iterative_normal2_gt,
            placeholder_flow2_gt: iterative_flow2_gt,
            placeholder_rotation_gt: iterative_rotation_gt,
            placeholder_translation_gt: iterative_translation_gt
        }

        # Backpropagate errors
        _, train_loss, train_summary = session.run(
            [optimizer, total_loss, total_loss_summary], feed_dict=feed_dict)
        writer_t.add_summary(train_summary, step)

        # Pop end from built-up iterative inputs (remove inputs that have been iterated on 4x)
        if iterative_image_pair_.shape[0] == 4 * batch_size:
            # final_image_pair_ = iterative_image_pair_[-batch_size:, :, :, :]
            # final_image2_2_ = iterative_image2_2_[-batch_size:, :, :, :]
            # final_depth2_ = iterative_depth2_[-batch_size:, :, :, :]
            # final_normal2_ = iterative_normal2_[-batch_size:, :, :, :]
            # final_rotation_ = iterative_rotation_[-batch_size:, :, :, :]
            # final_translation_ = iterative_translation_[-batch_size:, :, :, :]

            # Remove the final iteration from the total iteration data since it's been iterated 4x
            iterative_image_pair_ = iterative_image_pair_[0:-batch_size, :, :, :]
            iterative_image2_2_ = iterative_image2_2_[0:-batch_size, :, :, :]
            iterative_depth2_ = iterative_depth2_[0:-batch_size, :, :, :]
            iterative_normal2_ = iterative_normal2_[0:-batch_size, :, :, :]
            iterative_rotation_ = iterative_rotation_[0:-batch_size, :, :, :]
            iterative_translation_ = iterative_translation_[0:-batch_size, :, :, :]
            iterative_depth2_gt = iterative_depth2_gt[0:-batch_size, :, :, :]
            iterative_normal2_gt = iterative_normal2_gt[0:-batch_size, :, :, :]
            iterative_flow2_gt = iterative_flow2_gt[0:-batch_size, :, :, :]
            iterative_rotation_gt = iterative_rotation_gt[0:-batch_size, :, :, :]
            iterative_translation_gt = iterative_translation_gt[0:-batch_size, :, :, :]

        if step % 5000 == 0:
            save_path = saver.save(session, log_dir)
            print("Model saved in file: %s at step %d" % (save_path, step))
