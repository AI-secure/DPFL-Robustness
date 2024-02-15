# Folder Structure
After gathering all results, the folders can be structured as follows:
```
    saved_models
    ├── cifar_userdp_bkd            # backdoor attacks on CIFAR-10
    │   ├── noise1.5                # vary noise for different DP budget epsilon 
    │   │   ├── adv0                # vary k
    │   │   ├── adv1  
    │   │   ├── adv2     
    │   │   └── ...    
    │   ├── noise1.8     
    │   ├── noise2.0   
    │   ├── ........  
    │   ├── noise1.5_gamma1         # vary gamma
    │   │   ├── adv0     
    │   │   ├── adv1  
    │   │   ├── adv2     
    │   │   └── ...    
    │   ├── noise1.5_gamma100
    │   ├── noise1.5_alpha50        # vary poison ratio alpha
    │   ├── ........  
    │   ├── noise1.5_dba            # distributed backdoor
    │   └── ... 
    ├── mnist_userdp_bkd            # backdoor attacks on MNIST                     
    │   ├── noise1.3  
    │   │   ├── adv0     
    │   │   ├── adv1  
    │   │   ├── adv2     
    │   │   └── ...    
    │   ├── noise1.6    
    │   ├── ...
    │   ├── noise2.5_gamma1       
    │   └── ...            
    ├── cifar_insdp_bkd             # instance level DP
    ├── mnist_insdp_bkd
    ├── cifar_userdp_flip           # label flipping attacks 
    ├── mnist_userdp_flip 
    ├── cifar_insdp_flip
    ├── mnist_insdp_flip
    └── cifar100_model_best.pt.tar # the pretrained model on CIFAR100
```