import numpy as np

def score(actual,predicted):
    assert len(actual)==len(predicted)

    all=[]
    for i in range(len(actual)):
        all.append([actual[i],predicted[i]])

    sorted_all=sorted(all,key=lambda x:x[1],reverse=True)
    actual_sum=np.sum(sorted_all,axis=0)[0]
    current_item=0
    sum_item=0
    for item in sorted_all:
        current_item+=item[0]/(actual_sum+0.0)
        sum_item+=current_item
    sum_item-=(len(actual)+1)/2.0

    return sum_item/len(actual)



def normalized_score(actual,predicted):
    return score(actual,predicted)/score(actual,actual)


# print normalized_score([40, 0, 20, 0, 10], [1000000, 40, 40, 5, 5])