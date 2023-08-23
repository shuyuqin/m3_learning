from tqdm import tqdm
import numpy as np
import h5py

class STEM4D_DataSet:
    
    def __init__(self, 
                 data_dir, 
                 background_weight=0.10, 
                 crop = ((28,228),(28,228)), 
                 transpose = (1, 0, 3, 2), 
                 background_intensity = None,
                 counts_per_probe = 1e5,
                 rotation = None):
        """initial function of the class

        Args:
            data_dir (string): directory of the dataset
            background_weight (float, optional): _description_. Defaults to 0.10.
            crop (tuple, optional): _description_. Defaults to ((28,228),(28,228)).
            transpose (tuple, optional): _description_. Defaults to (1, 0, 3, 2).
            background_intensity (_type_, optional): _description_. Defaults to None.
            counts_per_probe (_type_, optional): _description_. Defaults to 1e5.
            rotation (_type_, optional): _description_. Defaults to None.
        """
        
        self.data_dir = data_dir
        self.background_weight = background_weight
        self.crop = crop
        self.x_size = self.crop[0][1]-self.crop[0][0]
        self.y_size = self.crop[1][1]-self.crop[1][0]
        self.transpose = transpose
        self.load_data()
        self.background_intensity = background_intensity
        self.counts_per_probe = counts_per_probe
        self.rotation = rotation
        if self.background_intensity is not None:
            self.generate_background_noise(self.stem4d_data,self.background_weight,self.counts_per_probe)

        if rotation is not None:
            self.rotate_data(self.stem4d_data,self.rotation)
        
    def load_data(self):
        if self.data_dir.endswith('.h5') or self.data_dir.endswith('.mat'):
            print(self.data_dir)
            with h5py.File(self.data_dir,'r') as f:
                stem4d_data = f['output4D']
                self.format_data(stem4d_data)
                
        elif self.data_dir.endswith('.npy'):
            self.stem4d_data = np.load(self.data_dir)
            self.format_data(stem4d_data)
            
    def format_data(self, stem4d_data):
        stem4d_data = stem4d_data[:, :, self.crop[0][0]
            :self.crop[0][1], self.crop[1][0]:self.crop[1][1]]
        stem4d_data = np.transpose(stem4d_data, self.transpose)
        stem4d_data = stem4d_data.reshape(-1, self.x_size, self.y_size)
        self.stem4d_data = stem4d_data
        
    def generate_background_noise(self,stem4d_data,background_weight,counts_per_probe):

        if background_weight == 0:
        
            self.stem4d_data = stem4d_data*1e5/4
            
            self.stem4d_data = self.stem4d_data.reshape(-1,1,self.x_size,self.y_size)

        else:
            noisy_data = np.zeros(stem4d_data.shape)
            im=np.zeros(stem4d_data.shape[1:])

            
            for i in tqdm(range(stem4d_data.shape[0]),leave=True,total=stem4d_data.shape[0]):
                test_img = np.copy(stem4d_data[i])
                qx = np.fft.fftfreq( im.shape[0], d = 1)
                qy = np.fft.fftfreq( im.shape[1], d = 1)
                qya, qxa = np.meshgrid(qy, qx)
                qxa = np.fft.fftshift(qxa)
                qya = np.fft.fftshift(qya) 
                qra2 = qxa**2 + qya**2
                im_bg = 1./( 1 + qra2 / 1e-2**2 )
                im_bg = im_bg / np.sum(im_bg) 
                int_comb = test_img * (1 - background_weight) + im_bg * background_weight 
                int_noisy = np.random.poisson(int_comb * counts_per_probe) / counts_per_probe
                int_noisy = int_noisy*1e5/4
                noisy_data[i] = int_noisy

            self.stem4d_data = noisy_data.reshape(-1,1,self.x_size,self.y_size)
        
    @property
    def stem4d_data(self):
        return self._stem4d_data
    
    @stem4d_data.setter
    def stem4d_data(self, stem4d_data):
        self._stem4d_data = stem4d_data
        
    def rotate_data(self, stem4d_data,rotation):
        
        self.angle = np.mod(np.arctan2(
                            rotation[:,1],
                            rotation[:,0]),
                            np.pi/3).reshape(-1)
        
        if self.angle.shape[0] != stem4d_data.shape[0]:
            print('the rotation size and image size do not match each other')

        else:
            # combine the data and label for test
            whole_data_with_rotation = []
            for i in tqdm(range(stem4d_data.shape[0]),leave=True, total=stem4d_data.shape[0]):
                whole_data_with_rotation.append([stem4d_data[i], self.angle[i]])
            
            self.stem4d_rotation = whole_data_with_rotation
        
        