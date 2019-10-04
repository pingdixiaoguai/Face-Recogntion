import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as img  # img 用于读取图片
import os  # 需要使用这个库的函数来看看TrainDatabase下有多少个图片
import numpy as np
"""
date:20190928
description:1.今天完成mat程序之将测试集转换为1维向量保存的程序（对应CreateDataBase）
            2.完成部分main函数（获取用户输入的test_image）
"""

"""
date:20190929
description:1.完成函数eigen_face_core，此函数返回：mean_of_train_database 训练集的平均值 (M*Nx1)矩阵
                                                 eigen_faces 训练数据库协方差矩阵的特征向量 (M*Nx(P-1))
                                                 centered_image_vectors 居中图像向量矩阵 (M*NxP)
"""

"""
date:20191001
description:1.完成所有工作
"""

"""
Author:pdxg
email:sylPDXG@qq.com
"""


def create_database(train_database_path='TrainDatabase/'):
    """
    description:此功能将训练数据库的所有2D图像整形为1D列向量。
                然后，将这些一维列向量放在一行中以构建二维矩阵“ T”。
    :param train_database_path: 训练集的路径,默认是同一目录下的TrainDatabase
    :return:一个2D矩阵，包含所有1D图像向量，
            假设训练数据库中的所有P图像具有相同的MxN大小。
            因此，一维列向量的长度为MN，“ T”将为MNxP 2D矩阵。
    """
    train_number = 0  # 记录训练图片数,应该是20
    ls = os.listdir(train_database_path)
    for i in ls:
        if os.path.isfile(os.path.join(train_database_path, i)):
            train_number += 1
    # print(train_number)  # 看看是不是20

    temp_array = []  # 此array用于暂时存放所有的一维数组

    for i in range(1, train_number + 1):
        train_image_path = 'TrainDatabase/' + str(i) + '.jpg'
        train_image = img.imread(train_image_path)  # 把图片以矩阵形式读入,此变量已经是一个矩阵形式了

        # print(train_image)  # 显示一下图片的矩阵形式（二维）
        # 先把获得的图片矩阵形式用numpy的array存储，之后转换成一维向量,MN * 1
        # 要注意的是matlab的reshape与python的不同，是行优先，所以要order = 'F'
        one_d_train_image = np.reshape(np.array(train_image), (-1, 1), order="F")

        # print(one_d_train_image)  # 显示一下每幅图片的对应一维向量
        # 之后把获得的这么多一维向量保存起来
        temp_array.append(one_d_train_image)

    # 将之前用于保存所有图片一维矩阵的array转换成numpy.matrix，用numpy.array好像有问题
    # 而且直接保存的这个矩阵不知为什么变成了P * MN的，必须转置回来
    one_d_train_image_all_set = np.transpose(np.mat(np.array(temp_array)))
    # print(one_d_train_image_all_set)  # 显示一下看看对不对
    # print(np.shape(one_d_train_image_all_set))
    return one_d_train_image_all_set


def eigen_face_core(one_d_train_image_all_set):
    """
    ：description 该函数获得一个包含所有训练图像矢量的2D矩阵，
                  并返回从训练数据库中提取的3个输出。
    :param one_d_train_image_all_set: 就是之前create_database创建出来的
                                      一个2D矩阵，包含所有1D图像向量，
                                      假设训练数据库中的所有P图像具有相同的MxN大小。
                                      因此，一维列向量的长度为M * N，此输入矩阵将为MNxP 2D矩阵。
    :return: mean_of_train_database 训练集的平均值 (M*Nx1)矩阵
             eigen_faces 训练数据库协方差矩阵的特征向量 (M*Nx(P-1))
             centered_image_vectors 居中图像向量矩阵 (M*NxP)
    """
    # 求出每个矩阵第m个的平均值（因为传入的是一系列一维的向量，所以是压缩列，变成一个mn * 1）的矩阵
    # 如果是压缩行，就是1 * mn的了
    mean_of_train_database = np.mat(np.mean(one_d_train_image_all_set, 1))
    # print(mean_of_train_database)  # 看看这个平均值矩阵
    # print(np.shape(mean_of_train_database))  # 看看它是不是MN * 1的,MN分别是256 x 384

    train_number = one_d_train_image_all_set.shape[1]  # 获取列数
    # print(train_number)  # 应该是20

    # 接下来求每个图像与平均图像的偏差,
    # 思路与create_datebase函数那个保存一堆一维向量的差不多,不在赘述。
    centered_image_vectors_temp = []
    for i in range(train_number):
        temp = one_d_train_image_all_set[:, i] - mean_of_train_database
        # print(temp)  # 看看每个分量的差
        centered_image_vectors_temp.append(temp)

    # 还是老问题求出来的会被转置，转置回去。
    centered_image_vectors = np.transpose(np.mat(np.array(centered_image_vectors_temp)))
    # 看看最后求出来这个偏差的对不对
    # print(centered_image_vectors)
    # print(np.shape(centered_image_vectors))  # 它应该是一个MN * P的矩阵

    """
    从线性代数理论我们知道，对于一个PxQ矩阵，
    该矩阵可以具有的非零特征值的最大数量为min（P-1，Q-1），
    由于训练图像的数量（P）通常小于像素的数量（M * N），因此可以找到的最大不为零的特征值等于P-1。    
    因此，我们可以计算A'* A（一个PxP矩阵）而不是A * A'（一个M * NxM * N矩阵）的特征值。 
    显然，A * A'的尺寸比A'* A大得多。 因此，维数将减小。（其实我没懂）
    """
    # 求一下特征值和特征向量，本来应该是MN * MN大小的，但是根据上面的说明，可以倒过来算
    covariance_matrix_temp = np.dot(np.transpose(centered_image_vectors), centered_image_vectors)
    # print(np.shape(covariance_matrix_temp))  # 应该是一个P * P大小的矩阵

    # 求特征向量和特征值，对covariance_matrix_temp的特征值也是那个A * A'的特征值
    # 特征值保存在一个数组里，而不像matlab一样是一个对角矩阵存储的，所以接下来的操作有所不同
    eigenvalues, feature_vector = np.linalg.eig(covariance_matrix_temp)
    # print(eigenvalues)  # 看看特征值,只有一个小于1,第8个（1开始）
    # print(np.transpose(feature_vector))  # 看看特征向量
    # print(np.shape(feature_vector))

    """
    对矩阵covariance_matrix_temp的所有特征值进行排序，并消除那些小于指定阈值的特征值。 
    因此，非零特征向量的数量可能少于（P-1）。
    """
    eigen_vector = []
    for i in range(feature_vector.shape[1]):
        if eigenvalues[i] > 1:
            eigen_vector.append(feature_vector[:, i])
    # 它又按照一行一行的存储了
    # print(eigen_vector)
    eigen_vector = np.transpose(np.mat(np.array(eigen_vector)))
    # print(np.shape(eigen_vector))
    # print(eigen_vector)

    """
    计算协方差矩阵A * A'的特征向量。
    可以从covariance_matrix_temp的特征向量中恢复协方差矩阵A * A'的特征向量。
    """
    eigen_faces = np.dot(centered_image_vectors, eigen_vector)
    # print(eigen_faces)

    return mean_of_train_database, eigen_faces, centered_image_vectors


def recognition(test_image, mean_of_train_database, eigen_faces, centered_image_vectors):
    """
    ：description：此功能通过将图像投影到面部空间并测量它们之间的欧式距离来比较两个面部。
    :param test_image: 输入图片
    :param mean_of_train_database: 训练集的平均值 (M*Nx1)矩阵
    :param eigen_faces: 训练数据库协方差矩阵的特征向量 (M*Nx(P-1))
    :param centered_image_vectors: 居中图像向量矩阵 (M*NxP)
    :return: output_name：训练数据库中识别图像的名称。
    """

    """
    将中心图像矢量投影到面部空间中，
    通过将本征面基数相乘将所有中心图像投影到面部空间中。 
    每个面部的投影矢量将是其对应的特征矢量。
    """
    projected_image = []
    train_number = eigen_faces.shape[1]
    for i in range(train_number):
        temp = np.dot(np.transpose(eigen_faces), centered_image_vectors[:, i])
        projected_image.append(temp)
    projected_image = np.transpose(np.mat(np.array(projected_image)))

    """
    从测试图像中提取PCA特征
    """
    in_image = np.reshape(np.array(test_image), (-1, 1), order="F")

    # 看看输入与平均m的区别，中心测试图像
    difference = in_image - mean_of_train_database

    # 测试图特征向量
    projected_test_image = np.dot(np.transpose(eigen_faces), difference)

    """
    计算欧几里得距离计算投影的测试图像与所有中心训练图像的投影之间的欧几里距离。 
    假设测试图像与训练数据库中的相应图像具有最小距离。
    """
    euclidean_distance = []
    for i in range(train_number):
        q = projected_image[:, i]
        temp = np.linalg.norm(projected_test_image - q) ** 2
        euclidean_distance.append(temp)
    euclidean_distance = np.transpose(np.mat(np.array(euclidean_distance)))
    # print(np.shape(euclidean_distance))

    """
    最后一步，把每行的最小值的行号存到recognized_index里
    注意下标从0，但是图片从1开始，要加1
    """
    euclidean_distance = np.array(euclidean_distance)
    euclidean_distance_min_index = np.argmin(euclidean_distance) + 1

    output_name = str(euclidean_distance_min_index) + '.jpg'
    return output_name


def main():
    """
    description:主函数
    :return: 无
    """
    # python没有do-while
    image_order = input('Enter test image name (a number between 1 to 10, q to quit): ')  # 获取需要被识别的那张图片
    if image_order == 'q':
        quit()
    num = int(image_order)
    while num < 1 or num > 10:
        print('Illegal input,please input again')
        image_order = input('Enter test image name (a number between 1 to 10, q to quit): ')
        if image_order == 'q':
            quit()
        num = int(image_order)
    test_image_path = 'TestDatabase/' + str(image_order) + '.jpg'  # 将获取到的图片路径拼接出来
    test_image = img.imread(test_image_path)  # 将图片读入

    plt.imshow(test_image)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.title('Input of PCA-Based Face Recognition System')  # 标题
    plt.show()  # 显示要被识别的图片

    one_d_train_image_all_set = create_database()
    mean_of_train_database, eigen_faces, centered_image_vectors = eigen_face_core(one_d_train_image_all_set)
    output_name = recognition(test_image, mean_of_train_database, eigen_faces, centered_image_vectors)

    selected_image = 'TrainDatabase/' + output_name
    selected_image = img.imread(selected_image)

    plt.imshow(selected_image)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.title('Equivalent Image')  # 标题
    plt.show()  # 显示要被识别的图片

    print('Matched image is: ' + output_name)


if __name__ == '__main__':
    while 1:
        main()
