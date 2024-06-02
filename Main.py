import numpy as np
import matplotlib.pyplot as plt
import cv2

figure_1 = np.array([[0.5, -1], [1, 0], [0, 0], [0.5, 1], [0.5, -1]], dtype=np.float32)
figure_2 = np.array([[1, 0], [0, 0], [1, 0.5], [0.5, 0.5], [1, 1], [1, 0]])
figure_3d = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0], [0, 0, 0], [0.5, 0.5, 1],
    [1, 0, 0], [1, 1, 0], [0.5, 0.5, 1], [0, 1, 0], [0.5, 0.5, 1]])
image_lab = cv2.imread('image_lab.jpg')
image_height, image_width, channels = image_lab.shape

angle = 45
angle_rad = np.radians(45)
center = (0.5, 0.5)
scale = 2
reflection_axis = "y"
shear_axis = "x"
shear_coefficient = 3
rotation_axis_3d = "z"
custom_matrix = np.array([[0, 0.3], [1, 0.4]])
custom_matrix_opencv = np.array([[0, 0.3, 4], [1, 0.4, 7]])


def print_figure_2d(figure):
    x_list = [coordinates[0] for coordinates in figure]
    y_list = [coordinates[1] for coordinates in figure]
    plt.plot(x_list, y_list)
    plt.show()


def print_figure_3d(figure):
    x_list = [coordinates[0] for coordinates in figure]
    y_list = [coordinates[1] for coordinates in figure]
    z_list = [coordinates[2] for coordinates in figure]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_list, y_list, z_list, 'b-', label='3D Pyramid')
    ax.legend()

    plt.show()


def rotate_figure(figure, angle_arg):
    rotation_matrix = np.array([[np.cos(angle_arg), -np.sin(angle_arg)], [np.sin(angle_arg), np.cos(angle_arg)]])

    rotated_figure = np.dot(figure, rotation_matrix)

    print_figure_2d(rotated_figure)


def rotate_figure_3d(figure, angle_arg, rotation_axis_3d_arg):
    rotation_matrix_x = np.array([[1, 0, 0], [0, np.cos(angle_arg), -np.sin(angle_arg)],
                                  [0, np.sin(angle_arg), np.cos(angle_arg)]])
    rotation_matrix_y = np.array([[np.cos(angle_arg), 0, np.sin(angle_arg)], [0, 1, 0],
                                  [-np.sin(angle_arg), 0, np.cos(angle_arg)]])
    rotation_matrix_z = np.array([[np.cos(angle_arg), -np.sin(angle_arg), 0],
                                  [np.sin(angle_arg), np.cos(angle_arg), 0], [0, 0, 1]])

    if rotation_axis_3d_arg == "x":
        rotated_figure_3d = np.dot(figure, rotation_matrix_x)
    elif rotation_axis_3d_arg == "y":
        rotated_figure_3d = np.dot(figure, rotation_matrix_y)
    elif rotation_axis_3d_arg == "z":
        rotated_figure_3d = np.dot(figure, rotation_matrix_z)

    print_figure_3d(rotated_figure_3d)


def scale_figure(figure, scale_arg):
    scale_matrix = np.array([[scale_arg, 0], [0, scale_arg]])

    scaled_figure = np.dot(figure, scale_matrix)

    print_figure_2d(scaled_figure)


def scale_figure_3d(figure, scale_arg):
    scale_matrix = np.array([[scale_arg, 0, 0], [0, scale_arg, 0], [0, 0, scale_arg]])

    scaled_figure = np.dot(figure, scale_matrix)

    print_figure_3d(scaled_figure)


def reflect_figure(figure, axis):
    reflection_matrix_x = np.array([[1, 0], [0, -1]])
    reflection_matrix_y = np.array([[-1, 0], [0, 1]])
    reflection_matrix_xy = np.array([[0, 1], [1, 0]])

    if axis == "x":
        reflected_figure = np.dot(figure, reflection_matrix_x)
    elif axis == "y":
        reflected_figure = np.dot(figure, reflection_matrix_y)
    elif axis == "y=x":
        reflected_figure = np.dot(figure, reflection_matrix_xy)

    print_figure_2d(reflected_figure)


def custom(figure, custom_matrix_arg):
    custom_figure = np.dot(figure, custom_matrix_arg)

    print_figure_2d(custom_figure)


def shear(figure, shear_coefficient_arg, shear_axis_arg):
    shear_matrix_x = np.array([[1, shear_coefficient_arg], [0, 1]])
    shear_matrix_y = np.array([[1, 0], [shear_coefficient_arg, 1]])

    if shear_axis_arg == "x":
        sheared_figure = np.dot(figure, shear_matrix_x)
    elif shear_axis_arg == "y":
        sheared_figure = np.dot(figure, shear_matrix_y)

    print_figure_2d(sheared_figure)


def rotate_opencv(figure, angle_arg, center_arg):
    rotation_matrix = cv2.getRotationMatrix2D(center_arg, angle_arg, 1)
    points = figure.reshape((-1, 1, 2))
    transformed_points = cv2.transform(points, rotation_matrix)
    transformed_matrix = transformed_points.reshape((-1, 2))
    print_figure_2d(transformed_matrix)


def scale_opencv(figure, center_arg):
    scaling_matrix = cv2.getRotationMatrix2D(center_arg, 0, 2)
    points = figure.reshape((-1, 1, 2))
    transformed_points = cv2.transform(points, scaling_matrix)
    transformed_matrix = transformed_points.reshape((-1, 2))
    print_figure_2d(transformed_matrix)


def reflect_opencv(figure, axis):
    reflection_matrix_x = np.array([[1, 0, 0], [0, -1, 0]])
    reflection_matrix_y = np.array([[-1, 0, 0], [0, 1, 0]])
    reflection_matrix_xy = np.array([[0, 1, 0], [1, 0, 0]])
    points = figure.reshape((-1, 1, 2))

    if axis == "x":
        transformed_points = cv2.transform(points, reflection_matrix_x)
    elif axis == "y":
        transformed_points = cv2.transform(points, reflection_matrix_y)
    elif axis == "y=x":
        transformed_points = cv2.transform(points, reflection_matrix_xy)

    transformed_matrix = transformed_points.reshape((-1, 2))
    print_figure_2d(transformed_matrix)


def custom_opencv(figure, custom_matrix_opencv_arg):
    points = figure.reshape((-1, 1, 2))
    transformed_points = cv2.transform(points, custom_matrix_opencv_arg)
    transformed_matrix = transformed_points.reshape((-1, 2))
    print_figure_2d(transformed_matrix)


def shear_opencv(figure, shear_coefficient_arg, shear_axis_arg):
    shear_matrix_x = np.array([[1, shear_coefficient_arg, 0], [0, 1, 0]])
    shear_matrix_y = np.array([[1, 0, 0], [shear_coefficient_arg, 1, 0]])
    points = figure.reshape((-1, 1, 2))

    if shear_axis_arg == "x":
        transformed_points = cv2.transform(points, shear_matrix_x)
    elif shear_axis_arg == "y":
        transformed_points = cv2.transform(points, shear_matrix_y)

    transformed_matrix = transformed_points.reshape((-1, 2))
    print_figure_2d(transformed_matrix)


def print_image(image, changed_image):
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('on')

    plt.subplot(1, 2, 2)
    plt.title('Changed Image')
    plt.imshow(cv2.cvtColor(changed_image, cv2.COLOR_BGR2RGB))
    plt.axis('on')

    plt.show()


def resize_image(image, image_width_arg, image_height_arg):
    new_width = scale * image_width_arg
    new_height = scale * image_height_arg
    scaled_image = cv2.resize(image, (new_width, new_height))

    print_image(image, scaled_image)


def rotate_image(image, angle_arg):
    rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle_arg, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (image.shape[1], image.shape[0]))

    print_image(image, rotated_image)
