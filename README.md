# Master Thesis
Code for master thesis: A Machine Learning accelerated geophysical fluid solver. 
Author: Yang Bai. 
Here we implement the classic and ML-based Euler and shallow water equation (SWE) solver. 
## Classic solvers. 
The classical solver contains 1d,2d Euler equations, 1d,2d, spherical SWEs equations solvers. They are implemented under [Dace][1] and [Torch][2] framework.  
### (1)Torch framework. 
Each solver contains a variety of interfaces, all of which can be entered using ArgumentParser, including resolution, order, run time, numerical flux, flux limiter, integration type, boundary condition and save format.  
Run it： **python SWE_sphere_torch.py --save_xdmf**. 
### (2)Dace framework. 
Due to the need for pre-compilation, some of the parameters, such as order and flux limiter could to be set manually.
#Validation. 
We use [Pyclaw][3] to compare with classic solvers. 
1. Reference solutions can use <u>sphere_reference.py</u>, <u>quad_reference</u> to generate. 
2. Here provided relative error <u>validation_error.py</u> and convergence order validation <u>validation_conver.py</u>. 

## ML-based solvers. 
The classical solver include three different CNN approach for 1d,2d and spherical SWEs equations solvers.  
1. If needed, the training set can be regenerated using “dataset_generator.py” and trained with the corresponding neural network model "cnn.py."  
2. test.ckpt test-16.ckpt etc. is the data of the trained model, you can set the solver type "classic" or "cnn" in ArgumentParser.  
e.g. **python SWE_sphere.py --solver=cnn**. 
In addition, you can also test existing models with the corresponding test files.   
e.g. **python SWE_sphere_test.py**. 
3.In the second and third solvers, you can additionally set the solver scale in ArgumentParser **--scale==8** **--scale==16** ...  

[1]:https://github.com/spcl/dace
[2]:https://pytorch.org/
[3]:https://github.com/clawpack/pyclaw
