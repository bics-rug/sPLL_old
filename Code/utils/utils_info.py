import numpy as np

def calculate_information(orientation_array,spatial_code):
    orientation_value = [orientation_array[i][1] for i in range(0, len(orientation_array))]
    list_orientation = np.unique(orientation_value)
    occurances = np.zeros([len(list_orientation), len(np.unique(spatial_code))])
    for row, orientation in enumerate(list_orientation):
        for column, pattern in enumerate(np.unique(spatial_code)):
            for trial in range(0, len(orientation_value)):
                if spatial_code[trial] == pattern and orientation_value[trial] == orientation:
                    occurances[row, column] += 1

    joint_prob = occurances / sum(sum(occurances))
    joint_prob = joint_prob + np.finfo(float).eps
    array_size = np.shape(joint_prob)
    marg_1 = joint_prob.sum(axis=1)
    marg_2 = joint_prob.sum(axis=0)
    information = 0
    for row in range(0, array_size[0]):
        for column in range(0, array_size[1]):
            dinformation = joint_prob[row, column] * np.log2(
                joint_prob[row, column] / (marg_1[row] * marg_2[column]))
            if np.isnan(dinformation):
                dinformation = 0
            information = dinformation + information
    #
    print("Information: " + str(
        round(information, 2)) + " bits. Recognized orientations:" + str(int(round(2 ** information))))
    print(
        "Number of orientation appeared: " + str(len(list_orientation)) + ".Number of patterns appeared: " + str(
            len(np.unique(spatial_code))))
    return information
def calculate_information_param(orientation_array,spatial_code):
    orientation_value = [orientation_array[i][1] for i in range(0, len(orientation_array))]
    list_orientation = unique(orientation_value)
    occurances = zeros([len(list_orientation), len(unique(spatial_code))])
    for row, orientation in enumerate(list_orientation):
        for column, pattern in enumerate(unique(spatial_code)):
            for trial in range(0, len(orientation_value)):
                if spatial_code[trial] == pattern and orientation_value[trial] == orientation:
                    occurances[row, column] += 1

    joint_prob = occurances / sum(sum(occurances))
    joint_prob = joint_prob + np.finfo(float).eps
    array_size = shape(joint_prob)
    marg_1 = joint_prob.sum(axis=1)
    marg_2 = joint_prob.sum(axis=0)
    information = 0
    for row in range(0, array_size[0]):
        for column in range(0, array_size[1]):
            dinformation = joint_prob[row, column] * log2(
                joint_prob[row, column] / (marg_1[row] * marg_2[column]))
            if isnan(dinformation):
                dinformation = 0
            information = dinformation + information
    #
    return information