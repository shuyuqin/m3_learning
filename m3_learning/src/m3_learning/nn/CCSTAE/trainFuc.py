import torch
from torch.utils.data import DataLoader
from .dataProcess import*
import numpy as np
import warnings

# fix seed of the training 



def Train(epochs=1000,scheduler=None,
          coef=0, ln_parm=1, mask_=mask_list, epoch_ = None,
          initial_epoch=120, file_path=None, best_train_loss= None):
    
         
        seed = 42
        torch.manual_seed(seed)
        train_iterator = DataLoader(noisy_data, batch_size=64, shuffle=True,num_workers=0)
#        print(config["learning_rate"])
        learning_rate = config["learning_rate"]
        
        coef_1 = config["l1_regular"]
        coef_2 = config["scale_regular"]
        coef_3 = config["shear_regular"]
        coef_4 = config["custom_para"]
        
        
        encoder, decoder, join, optimizer = make_model(learning_rate = learning_rate)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=3e-4,
                                              step_size_up=50,cycle_momentum=False)
        
        
#         for g in optimizer.param_groups:
#             g['lr'] = learning_rate
        
        N_EPOCHS = epochs

        if best_train_loss == None:
            best_train_loss = float('inf')


        if epoch_==None:
            start_epoch = 0
        else:
            start_epoch = epoch_+1

        for epoch in range(start_epoch,N_EPOCHS):
    #    This loss function include the entropy loss with increasing coefficient value

            train = loss_function_epoch(join,train_iterator,
                                      optimizer,coef_1,coef_2,coef_3,coef_4,ln_parm,mask_)
                
            train_loss,L2_loss,Scale_Loss, Shear_Loss = train

            train_loss /= len(train_iterator)
            L2_loss /= len(train_iterator)
            Scale_Loss /= len(train_iterator)
            Shear_Loss /= len(train_iterator)

    #        VAE_L /= len(train_iterator)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
    #        print(f'......... VAE Loss: {VAE_L:.4f}')
            print('.............................')
            
            wandb.log({"epcoh": epoch,
                       "l1_loss": L2_loss,
                       "scale_loss": Scale_Loss,
                       "shear_loss": Shear_Loss,
                       "train_loss": train_loss, 
                      })
          #  schedular.step()
            if best_train_loss > train_loss:
                best_train_loss = train_loss
                patience_counter = 1
                checkpoint = {
                    "net": join.state_dict(),
                    "encoder":encoder.state_dict(),
                    "decoder":decoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    "epoch": epoch,
                    'mse_loss': train_loss,
                }
                if epoch >=10:
                    lr_ = format(optimizer.param_groups[0]['lr'],'.5f')
#                    l1_form = format(coef_1,'.4f')
                    scale_form = format(coef_2,'.4f')
                    shear_form = format(coef_3,'.4f')
                    cust_form = format(coef_4,'0d')
                    file_path = 'TEST_automation_code_2_27/NOISY_25percent_02_19_lr:'+lr_+'_scale_cof:'+scale_form+'_shear_cof:'+shear_form+'_custom_para:'+cust_form+\
                               f'_epoch:{epoch:04d}_trainloss:{train_loss:.5f}_l1:{L2_loss:.5f}_scal:{Scale_Loss:.5f}_shr:{Shear_Loss:.5f}.pkl'
                    torch.save(checkpoint, file_path)
            if epoch==999:
                del join
                del encoder
                del decoder
            if scheduler!= None:
                scheduler.step()



def add_kwargs(obj, **kwargs):
    for key, value in kwargs.items():
        if not hasattr(obj, key):
            warnings.warn(f"Attribute '{key}' does not exist in the object. Setting it now.")
        setattr(obj, key, value)


class Train_Class:



    def __init__(self,
                 data_dir,
                loss_class,
                 background_weight = 0.25,
                 background_intensity = None,
                 rotation = None,
                 crop = ((28,228),(28,228)), 
                 transpose = (1, 0, 3, 2), 
                 counts_per_probe = 1e5,
                 seed = 42,
                 device = 'cpu',
                 regular_norm='l1',
                 mask_list= None,
                 mask_for_updating = None,
                 Up_inp = False,
                 file_path = None,
                 folder_path='',
                 set_scheduler = False,
                 lr_decay = False, 
                learning_rate=3e-5,
                max_rate = 2e-4,
                l_norm_coef = 1e-6,
                scale_reg = 1e-2,
                shear_reg = 1e-2,
                batch_para = 16,
                epochs = 20,
                radius = 45,
                batch_size=64,
                 ):
    
        self.regular_norm =regular_norm
        self.device = device
        self.seed = seed
        self.mask_list = mask_list 
        self.mask_for_updating = mask_for_updating
        self.interpolate = Up_inp
        self.fle_path = file_path
        self.folder_path = folder_path
        self.set_scheduler = set_scheduler
        self.lr_decay = lr_decay
        self.learning_rate = learning_rate
        self.max_rate = max_rate
        self.l_norm_coef = l_norm_coef
        self.scale_reg = scale_reg


        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        load_data_class = STEM4D_DataSet(data_dir,
                                       background_weight,
                                       crop,
                                       transpose, 
                                       background_intensity,
                                       counts_per_probe,
                                       rotation)
        if rotation is not None:
            self.data_set = load_data_class.stem4d_rotation
        else:
            self.data_set = load_data_class.stem4d_data
    


    def train_model(self,
                **kwargs
                    ):

        add_kwargs(self, **kwargs)

        # fix seed of the model
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)


        learning_rate = int(self.learning_rate*1e6)/1e6
        max_rate = int(self.ax_rate*1e6)/1e6
        coef_1 = int(self.l_norm_coef*1e9)/1e9
        coef_2 = int(self.scale_reg*1e2)/1e2
        coef_3 = int(self.shear_reg*1e2)/1e2
        
        patience = 0
        
        print("........successfully load parameters")
        
        
        encoder, decoder, join, optimizer = \
        make_model_2(device,learning_rate = learning_rate,fixed_mask = mask_)
        
        radius = 45
        
            
        if set_scheduler:
        
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=max_rate,
                                                  step_size_up=20,cycle_momentum=False)
        else: 
            
            lr_scheduler = None
        

        
        if self.regular_norm == 'l1':
            regul_type = 1
            
        elif self.regular_norm == 'l2':
            regul_type = 2
            
        else:
            regul_type = 0
            
     
        print('..........successfully generate model')
        
        
        train_iterator = DataLoader(self.data_set, batch_size=batch_size, shuffle=True, num_workers=0)
#        converge_iterator = DataLoader(whole_data_with_rotation, batch_size=8, shuffle=True, num_workers=0)
        
        test_iterator = DataLoader(self.data_set, batch_size=batch_size, shuffle=False, num_workers=0)

        N_EPOCHS = epochs

    
        best_train_loss = float('inf')

        print('...........successfully generate train interator')

        for epoch in range(0,epochs):
    #    This loss function include the entropy loss with increasing coefficient value
    
            if epoch >0 and epoch<2:
            
                encoder, decoder, join, optimizer = \
                        make_model_2(device,learning_rate = learning_rate,fixed_mask = mask_)
                
                
                check_ccc = torch.load(file_path) 

                join.load_state_dict(check_ccc['net'])
                encoder.load_state_dict(check_ccc['encoder'])
                decoder.load_state_dict(check_ccc['decoder'])
                optimizer.load_state_dict(check_ccc['optimizer'])

                
            optimizer.param_groups[0]['lr'] = learning_rate   
                

            train = loss_class.(join,train_iterator,
                                  optimizer,device,coef_1,coef_2,coef_3,regul_type,mask_,interpolate_2)

            input_length = len(train_iterator)
                
                
            train_loss,L2_loss,Scale_Loss, Shear_Loss = train
            
            train_loss /= input_length
            L2_loss /= input_length
            Scale_Loss /= input_length
            Shear_Loss /= input_length

    #        VAE_L /= len(train_iterator)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
    #        print(f'......... VAE Loss: {VAE_L:.4f}')
            print('.............................')
        
            
            name_of_file = folder_path + f'/L1:{coef_1:.10f}_scale:{coef_2:.3f}_shear:{coef_3:.3f}_lr:{learning_rate:.6f}_batchpara:{batch_para:02d}_Epoch:{epoch:04d}_trainloss:{train_loss:.6f}_'

            exx_Shuyu,eyy_Shuyu,exy_Shuyu = Show_Process(join,test_iterator,mask_,name_of_file,device,interpolate_2)
            
            if epoch==0:
        
                
                
                center_mask_list, rotate_center = \
                inverse_base(name_of_file, check_mask, radius = radius)

     #           rotation_ = load_rotation_

                mask_ = center_mask_list

            # Save every weight to be prepared for loading
    
            


            checkpoint = {
                "net": join.state_dict(),
                "encoder":encoder.state_dict(),
                "decoder":decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'mse_loss': train_loss,
            }
            if epoch >=0:
                lr_ = format(optimizer.param_groups[0]['lr'],'.6f')
#                    l1_form = format(coef_1,'.4f')
                scale_form = format(coef_2,'.3f')
                shear_form = format(coef_3,'.3f')
                file_path = folder_path+'/07_13_15D_upsample_update_mask_once_lr:'+lr_+'_scale_cof:'+scale_form+'_shear_cof:'+shear_form+\
                            f'_MAE:{combine_loss:.4f}_seed:{seed:02d}_epoch:{epoch:04d}_trainloss:{train_loss:.6f}_l1:{L2_loss:.5f}_scal:{Scale_Loss:.5f}_shr:{Shear_Loss:.5f}.pkl'
                torch.save(checkpoint, file_path)
                
            if epoch>=1:
            
                if best_train_loss > train_loss:
                    best_train_loss = train_loss

                    patience = 0

                    learning_rate = 1.2 * learning_rate

                else:
                    patience +=1

                    if patience >0:
                        learning_rate = learning_rate*0.8
                
            tune.report(loss=train_loss, score=combine_loss)

        if lr_scheduler!= None:
                lr_scheduler.step()






