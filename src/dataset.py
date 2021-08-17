from PIL import Image
from torch.utils.data import Dataset


#Create general retina dataset to support train,val loader
class RetinaDataset(Dataset):
    
  def __init__(self, folder_dir, dataframe, image_size, normalization=True):
    self.image_paths = [] 
    self.image_labels = [] 

    # Define list of image transformations
    image_transformation = [
        transforms.Resize(image_size),
        transforms.ToTensor()]

    if normalization:
        image_transformation.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))

    self.image_transformation = transforms.Compose(image_transformation)
    
    # Get all image paths and image labels from dataframe
    for index, row in dataframe.iterrows():
        image_path = os.path.join(folder_dir, row.filename)
        self.image_paths.append(image_path)
        labels = []
        for col in row[1:]:
            if col == 1:
                labels.append(1)
            else:
                labels.append(0)

        self.image_labels.append(labels)
            
  def __len__(self):
      return len(self.image_paths)

  def __getitem__(self, index):
      # Read image
      image_path = self.image_paths[index]
      image_data = Image.open(image_path).convert("RGB") # Convert image to RGB channels

      # TODO: Image augmentation code would be placed here
      #...
      image_data = self.image_transformation(image_data)

      return image_data, torch.FloatTensor(self.image_labels[index]),image_path