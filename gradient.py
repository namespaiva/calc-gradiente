# import numpy as np
# import sys
# sys.setrecursionlimit(10000)

# def grad(input,output, w1, w2, b):
#     dL_dw1 = -2 * sum((output - (w1*input**2 + w2*input + b)) * input**2)
#     dL_dw2 = -2 * sum((output - (w1*input**2 + w2*input + b)) * input)
#     dL_db = -2 * sum(output - (w1*input**2 + w2*input + b))
#     return dL_dw1,dL_dw2,dL_db

# def grad2(input, output, w1, b):
#     dl_dw = -2 * sum(input * (output - (w1 * input + b)))
#     dl_db = -2 * sum(output - w1 * input - b)
#     return dl_db, dl_dw

# def descent(init,lr,epochs,input,output,gradient,tol=10**-6):
#     count = 0
#     for _ in range (epochs):
#         count+=1
#         Mygrad = gradient(input,output,*init)
#         # a = init[0] - lr * a
#         # b = init[1] - lr * b
#         # c = init[2] - lr * c
#         Mygrad = [init[i]-lr*Mygrad[i] for i in range(len(Mygrad))]
#         err = np.sqrt(sum((Mygrad[i]-init[i])**2 for i in range(len(Mygrad)))) #(a-init[0])**2 + (b-init[1])**2 + (c-init[2])**2
#         if err < tol:
#             break
#         init = Mygrad

#         return Mygrad,err,count

# data = np.array([(1,4),(-1,5),(2,7),(-2,8),(3,12),(-3,13),(4,19),(-4,20)])
# x = data[:, 0]
# y = data[:, 1]

# print(descent((2,2,2),10**-10,10**5,x,y,grad))

import numpy as np
import sys
sys.setrecursionlimit(10000)

def grad(input,output, w1, w2, b):
    dL_dw1 = -2 * sum((output - (w1*input**2 + w2*input + b)) * input**2)
    dL_dw2 = -2 * sum((output - (w1*input**2 + w2*input + b)) * input)
    dL_db = -2 * sum(output - (w1*input**2 + w2*input + b))
    return dL_dw1,dL_dw2,dL_db

def grad2(input, output, w1, b):
    dl_dw = -2 * sum(input * (output - (w1 * input + b)))
    dl_db = -2 * sum(output - w1 * input - b)
    return dl_db, dl_dw

def descent(init,lr,epochs,input,output,gradient,tol=10**-6):
    count = 0
    for _ in range (epochs):
        count+=1
        Mygrad = gradient(input,output,*init)
        # a = init[0] - lr * a
        # b = init[1] - lr * b
        # c = init[2] - lr * c
        Mygrad = [init[i]-lr*Mygrad[i] for i in range(len(Mygrad))]
        err = sum((Mygrad[i]-init[i])**2 for i in range(len(Mygrad))) #(a-init[0])2 + (b-init[1])2 + (c-init[2])**2
        if err < tol:
            break
        init = Mygrad

    return Mygrad,err,count

    


data = np.array([(1,4),(-1,5),(2,7),(-2,8),(3,12),(-3,13),(4,19),(-4,20)])
x = data[:, 0]
y = data[:, 1]

print(descent((2,2,2),10**-3,10**5,x,y,grad))