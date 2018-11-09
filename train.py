# This is the main training script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

from argparse import ArgumentParser
import mycoco
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Dropout, Activation, Conv2DTranspose
from keras.models import Model
from keras.models import load_model
from keras.utils import plot_model

def optA():
    mycoco.setmode('train')
    # loading images
    # i've modified iter_images in mycoco.py to only return the images without the labels
    cat_list = []
    for cat in args.categories:
        cat_list.append([cat])
    n_classes = len(cat_list)
    allids = mycoco.query(cat_list)
    if args.maxinstances:
        imgs = mycoco.iter_images([x[:int(args.maxinstances)] for x in allids], [x for x in range(n_classes)], batch=16)
        n_imgs = int(args.maxinstances) * len(cat_list)
    else:
        imgs = mycoco.iter_images([x for x in allids], [x for x in range(n_classes)], batch=16)
        n_imgs = sum(len(x) for x in allids)

    # print(n_imgs)



    # model layers:

    # encoder
    input_img = Input(shape=(200, 200, 3))

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # I'm not sure here if i should extract the last maxpooling2d layer for the second model
    # or if there are also supposed to be flatten and dense layers before the extraction.
    # I've also tried making flatten and dense layers as the connecting step
    # between both parts in the autoencoder model to make the extracted bottleneck layer
    # 2 dimensional. I've since discarded those as I wasn't sure they were needed.

    # flat_encoded = Flatten()(encoded)
    # denselayer = Dense(n_classes, activation='softmax')(flat_encoded)

    # decoder
    x = Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2DTranspose(3, (3, 3), activation='softmax', padding='same')(x)

    # autoencoder model
    autoencoder = Model(input_img, decoded)
    autoencoder.summary()
    # encoder model
    encoder = Model(input_img, encoded)
    encoder.summary()
    # plot_model(autoencoder, to_file='autoencoder.png', show_shapes=True, show_layer_names=True)
    # plot_model(encoder, to_file='encoder.png', show_shapes=True, show_layer_names=True)
    autoencoder.compile(loss='mean_squared_error', optimizer='adam')

    batch = 16
    autoencoder.fit_generator(imgs, steps_per_epoch=(n_imgs / batch), epochs=20)

    encoder.set_weights(autoencoder.get_weights()[0:7])
    preds = encoder.predict_generator(imgs, steps=(n_imgs))

    print("Predictions shape: ", preds.shape)

    # saving the autoencoder model
    autoencoder.save(args.modelfile)



# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('train')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Train a model.")
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    # parser.add_argument('checkpointdir', type=str,
    #                     help="directory for storing checkpointed models and other metadata (recommended to create a directory under /scratch/)")
    parser.add_argument('modelfile', type=str, help="output model file")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    # print("Working directory at " + args.checkpointdir)
    print("Maximum instances is " + str(args.maxinstances))

    if len(args.categories) < 2:
        print("Too few categories (<2).")
        exit(0)

    print("The queried COCO categories are:")
    for c in args.categories:
        print("\t" + c)

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB()
    else:
        print("Option does not exist.")
        exit(0)
