# This is the main testing script that we should be able to run to grade
# your model training for the assignment.
# You can create whatever additional modules and helper scripts you need,
# as long as all the training functionality can be reached from this script.

from argparse import ArgumentParser
import mycoco
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Dropout, Activation, Conv2DTranspose
from keras.models import Model
from keras.models import load_model

def optA():
    mycoco.setmode('test')
    # loading images
    cat_list = []
    for cat in args.categories:
        cat_list.append([cat])
    n_classes = len(cat_list)
    allids = mycoco.query(cat_list)
    print("Creating image list...")
    if args.maxinstances:
        test_x, test_y = mycoco.image_list([x[:int(args.maxinstances)] for x in allids],
                                             [x for x in range(n_classes)])
    else:
        test_x, test_y = mycoco.image_list([x for x in allids], [x for x in range(n_classes)])
    imgs = img_gen(test_x)


    # loading the autoencoder model
    autoencoder = load_model(args.modelfile)

    # encoder
    input_img = Input(shape=(200, 200, 3))

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    encoder = Model(input_img, encoded)
    encoder.set_weights(autoencoder.get_weights()[0:7])
    preds = encoder.predict_generator(imgs, steps=(len(test_x)))

    print("Predictions shape: ", preds.shape)

def img_gen(img_list):
    while True:
        for i in img_list:
            yield (i, i)

# If you do option B, you may want to place your code here.  You can
# update the arguments as you need.
def optB():
    mycoco.setmode('test')
    print("Option B not implemented!")

# Modify this as needed.
if __name__ == "__main__":
    parser = ArgumentParser("Evaluate a model.")
    # Add your own options as flags HERE as necessary (and some will be necessary!).
    # You shouldn't touch the arguments below.
    parser.add_argument('-P', '--option', type=str,
                        help="Either A or B, based on the version of the assignment you want to run. (REQUIRED)",
                        required=True)
    parser.add_argument('-m', '--maxinstances', type=int,
                        help="The maximum number of instances to be processed per category. (optional)",
                        required=False)
    parser.add_argument('modelfile', type=str, help="model file to evaluate")
    parser.add_argument('categories', metavar='cat', type=str, nargs='+',
                        help='two or more COCO category labels')
    args = parser.parse_args()

    print("Output model in " + args.modelfile)
    print("Maximum instances is " + str(args.maxinstances))

    print("Executing option " + args.option)
    if args.option == 'A':
        optA()
    elif args.option == 'B':
        optB()
    else:
        print("Option does not exist.")
        exit(0)
