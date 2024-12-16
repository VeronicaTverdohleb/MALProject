# Define a class for extracting features using a pretrained CNN - Resnet50
class KerasFeatureExtractor:
    def __init__(self):
        # Initialize the ResNet50 model with pretrained ImageNet weights
        self.model = ResNet50(weights='imagenet', include_top=False, input_shape=(348, 348, 3))
        # include_top=False removes the fully connected layers at the top of the network which are used for classification
        # This configuration is chosen because the goals is to use the model as a feature extractor rather than
        # for its ability to classify the ImageNet classes directly
        
        # Set up the data generator for standard preprocessing and augmentation
        self.datagen = ImageDataGenerator(rescale=1.0/255)  # Rescale pixel values from [0,255] to [0,1]

    # Define a method to extract features from directories listed in a text file
    def extract_features_from_file(self, txt_file):
        # Read the text file and get the directory paths
        with open(txt_file, 'r') as f:
            directories = f.read().splitlines()  # Read each directory line by line

        all_features = []  # List to hold the extracted features
        all_labels = []  # List to hold the corresponding labels

        for directory in directories:
            # Clean up the directory path (strip unwanted characters)
            directory = 'GroceryStoreDataset-working/dataset/' + directory.split(',')[0].strip()  # Extract only the valid path, remove any extra data
            
            if not os.path.exists(directory):
                print(f"Directory not found: {directory}")
                continue  # Skip this directory if it doesn't exist

            # Configure the generator to read data from the directory
            generator = self.datagen.flow_from_directory(
                directory,
                target_size=(348, 348),  # Resize images to 348x348
                batch_size=32,  # Process images in batches of 32
                class_mode='categorical',  # Generate one-hot encoded labels
                shuffle=False  # Do not shuffle images to maintain label order
            )
            print(f"Processing directory: {directory}")
            print(generator.class_indices)

            features = []  # List to hold features for this directory
            labels = []  # List to hold labels for this directory

            # Loop over batches of images
            for batch_data, batch_labels in generator:
                # Predict features for batch using ResNet50
                batch_features = self.model.predict(batch_data)
                features.append(batch_features)  # Append features to the list
                print(f"Processed {len(features) * 32}/{generator.samples} images")  # Debugging output
                labels.append(batch_labels)  # Append labels to the list
                if len(features) * 32 >= generator.samples:
                    break

            all_features.append(np.vstack(features))  # Append all features for the directory
            all_labels.append(np.vstack(labels))  # Append all labels for the directory

        # Resnet50 outputs each image as a three-dimensional array (height x width x channels)
        # This step is used to convert the three-dimensional feature maps into to one-dimensional vectors
        # This step is crucial for compatibility with RandomForest and XGBoost classifiers
        return np.vstack(features).reshape(-1, np.prod(batch_features.shape[1:])), np.vstack(labels)

# Create an instance of the feature extractor
extractor = KerasFeatureExtractor()

# Extract features and labels for training and validation sets
train_features, train_labels = extractor.extract_features_from_file('GroceryStoreDataset-working/dataset/train.txt')
val_features, val_labels = extractor.extract_features_from_file('GroceryStoreDataset-working/dataset/val.txt')



# Extract features and labels for each fruit class separately
# Extract features for a list of image paths from a text file
def extract_features_from_file(txt_file, model, datagen):
    # Read the text file and get the image paths and labels
    with open(txt_file, 'r') as f:
        image_paths = f.read().splitlines()  # Read each line (path) from the file

    all_features = []  # List to hold the extracted features
    all_labels = []  # List to hold the corresponding labels

    for line in image_paths:
        # Clean up the path (strip unwanted characters)
        stripped_path = line.split(',')[0].strip()  # Extract only the valid image path
        path = os.path.join('GroceryStoreDataset-working/dataset', stripped_path)  # Create full path

        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue  # Skip this file if it doesn't exist

        # Load and preprocess the image
        img_array = load_and_preprocess_image(path, datagen)

        # Extract features using ResNet50
        features = model.predict(img_array)
        all_features.append(features)  # Append features to the list
        
        # Extract label (if your labels are stored in the text file in a certain way)
        label = int(line.split(',')[1].strip())  # Assuming labels are in the second part of the line
        all_labels.append(label)  # Append the label

    return np.vstack(all_features).reshape(-1, np.prod(features.shape[1:])), np.array(all_labels)

