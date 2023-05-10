# def countOperations( num1, num2):
#     t = 0
#     while True:
#         if (num1 >= num2):
#             num1 = num1 - num2
#             t = t + 1
#         else:
#             num2 = num2 - num1
#             t = t + 1
#         if(num1==0 or num2==0):
#             break
#     return t
def countOperations(self, num1, num2):
    t = 0
    while (num1!= 0 and num2 != 0):
        if (num1 >= num2):
            num1 = num1 - num2
            t = t + 1
        else:
            num2 = num2 - num1
            t = t + 1
    return t
print(countOperations("",3,5))