import os
import csv
import cv2
import numpy as np
import matplotlib.image as mpimg

class Utils():

    CSV_SRC = "driving_log.csv"
    IMG_SRC = "IMG"
    STEERING_COEFFICIENT = 0.229
    TOP_TRIM = 0.35
    BOTTOM_TRIM = 0.10
    DATASET_DIM = (64, 64)

    def next_batch(self, batch_size=128):
        # creating a generative function , using yield to get subsequent batches
        while True:
            X_images = []
            y_labels = []
            images = self.get_batch_files(batch_size)
            for img_file, angle in images:
                image = mpimg.imread(os.path.join(self.IMG_SRC, img_file))
                new_image, new_angle = self.process_image(image, angle)
                X_images.append(new_image)
                y_labels.append(new_angle)

            yield np.array(X_images), np.array(y_labels)

    def process_image(self, image, steering_angle):
        up = int(np.ceil(image.shape[0] * self.TOP_TRIM))
        down = image.shape[0] - int(np.ceil(image.shape[0] * self.BOTTOM_TRIM))
        image = image[up:down, :]
        image = cv2.resize(image, self.DATASET_DIM)

        return image, steering_angle


    def get_batch_files(self,batch_size=128):
        """
        Reads the CSV files and fetches a list of images and the steering angles
        :return:
        """
        lines = []
        with open(self.CSV_SRC) as csvfile:
            reader = csv.reader(csvfile)
            #cnt = 0
            for line in reader:
                lines.append(line)
                #if cnt == 5000:
                #    break
                #cnt += 1

        no_img = len(lines)

        # Randomizing the order of left, right and center images
        rd_idx = np.random.randint(0, no_img, batch_size)

        # list of tuples of image files with the corresponding steering angles
        img_tuple = []

        for idx in rd_idx:
            # randomly choosing the camera view
            rd_cam = np.random.randint(0, 3)

            # if left
            if rd_cam == 0:
                img = line[rd_cam].split('/')[-1]
            elif rd_cam == 1:
                img = line[rd_cam].split('/')[-1]
            else:
                img = line[2].split('/')[-1]

            angle = float(line[3]) + self.STEERING_COEFFICIENT
            img_tuple.append((img, angle))
        return img_tuple



"""
def get_data():
    lines = []
    SRC = os.path.join("BC")
    IMG_SRC = os.path.join(SRC,"IMG")
    CSV_SRC = os.path.join(SRC,"driving_log.csv")

    with open(CSV_SRC) as csvfile:
        reader = csv.reader(csvfile)
        cnt = 0
        for line in reader:
            lines.append(line)
            if cnt == 5000:
                break
            cnt += 1

    images = []
    measurements = []

    for line in lines:
        source_path = line[0]
        filename = source_path.split('/')[-1]
        image = mpimg.imread(os.path.join(IMG_SRC,filename))

        images.append(image)

        measurement = float(line[3])
        measurements.append(measurement)

    X_train = np.array(images)
    y_train = np.array(measurements)

    return X_train, y_train

"""
