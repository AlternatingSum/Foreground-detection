#!/usr/bin/env python2

"""
Foreground detection
"""
import skimage
from skimage import io
from skimage import color
from skimage.transform import resize
import math
import matplotlib.pyplot as plt
import numpy as np


def create_scaled_images(image):
    """Creates a list of scaled versions of the image"""
    height = len(image)
    width = len(image[0])
    largest = max(height, width)
    image = resize(image, (int(height*400/largest), int(width*400/largest)))
    
    scaled_images = [image]
    smallest_image = image

    while len(smallest_image) > 40 and len(smallest_image[0]) > 40:
        height = len(smallest_image)
        width = len(smallest_image[0])
        smallest_image = resize(smallest_image, (int(height/1.5), int(width/1.5)))
        scaled_images.append(smallest_image)
    
    return scaled_images


def mean_color(image):
    """Finds the mean color in the image, defined using RGB values"""
    
    height = len(image)
    width = len(image[0])
    num_pixels = height * width
    total_red = 0
    total_green = 0
    total_blue = 0
    
    for row in range(height):
        for col in range(width):
            total_red += image[row][col][0]
            total_green += image[row][col][1]
            total_blue += image[row][col][2]
    
    mean_red = float(total_red)/float(num_pixels)
    mean_green = float(total_green)/float(num_pixels)
    mean_blue = float(total_blue)/float(num_pixels)
    
    return [mean_red, mean_green, mean_blue]


def sigmas(image):
    """Finds the standard deviations of color in the image, in terms of hue / saturation / value"""
    
    mean = mean_color(image)
    height = len(image)
    width = len(image[0])
    num_pixels = height * width
    mean_hsv = color.convert_colorspace([[mean]], 'RGB', 'HSV')[0][0]
    image_hsv = color.convert_colorspace(image, 'RGB', 'HSV')
    total_hue = 0
    total_sat = 0
    total_val = 0
    
    for row in range(height):
        for col in range(width):
            """Note that the difference in hues does not use the standard metric, since the space of hues is circular"""
            hue_diff = min(abs(mean_hsv[0] - image_hsv[row][col][0]), 1 - abs(mean_hsv[0] - image_hsv[row][col][0]))
            sat_diff = abs(mean_hsv[1] - image_hsv[row][col][1])
            val_diff = abs(mean_hsv[2] - image_hsv[row][col][2])
            total_hue += hue_diff**2
            total_sat += sat_diff**2
            total_val += val_diff**2
    
    sigma_hue = math.sqrt(float(total_hue) / float(num_pixels))
    sigma_sat = math.sqrt(float(total_sat) / float(num_pixels))
    sigma_val = math.sqrt(float(total_val) / float(num_pixels))
    
    return [sigma_hue, sigma_sat, sigma_val]


def gaussian(distance, sigma):
    """Evaluates a non-normalized Gaussian distribution"""
    
    return math.exp(-distance**2/(2*sigma**2))


def find_m_values(image):
    """Determines the value of m which best fits a hyperbola to the histogram of color differences
    for each color parameter (hue, saturation, value)"""
    
    height = len(image)
    width = len(image[0])
    image_hsv = color.convert_colorspace(image, 'RGB', 'HSV')
    m_values = [0.0, 0.0, 0.0]
    x_list = range(20)
    y_lists = [[0]*20, [0]*20, [0]*20]
    
    for row in range(height):
        for col in range(width):
            current_color = image_hsv[row][col]
            for neighbor in forward_pixel_neighbors([row, col], height, width):
                new_color = image_hsv[neighbor[0]][neighbor[1]]
                """Note that the difference in hues does not use the standard metric, since the space of hues is circular"""
                hue_diff = min(abs(current_color[0] - new_color[0]), 1 - abs(current_color[0] - new_color[0]))
                sat_diff = abs(current_color[1] - new_color[1])
                val_diff = abs(current_color[2] - new_color[2])
                x_hue = int(100*hue_diff)
                x_sat = int(100*sat_diff)
                x_val = int(100*val_diff)
                if x_hue < 20:
                    y_lists[0][x_hue] += 1
                if x_sat < 20:
                    y_lists[1][x_sat] += 1
                if x_val < 20:
                    y_lists[2][x_val] += 1
    
    for parameter in range(3):
        y_list = y_lists[parameter]
        max_y_val = max(y_list)
        if max_y_val > 0.0:
            for index in range(len(y_list)):
                    y_lists[parameter][index] = float(y_lists[parameter][index])/float(max_y_val)
        m = fit_hyperbola(x_list, y_list, 0.1)
        m_values[parameter] = m
    
    return m_values


def hyperbola_cost(x_vals, y_vals, m):
    """Determines a cost to indicate how well a certain hyperbola fits the data"""
    
    cost = 0.0
    for index in range(len(x_vals)):
        x = x_vals[index]
        y = y_vals[index]
        
        if y != 0:
            outcome = hyperbola_prediction(x, m)
            cost += (y - outcome)**2
    
    return cost / len(x_vals)


def hyperbola_prediction(x, m):
    """Uses a hyperbola to predict an outcome"""
    
    x = float(x)
    m = float(m)
    
    return math.sqrt(1 + (m*x)**2) - m*x


def hyperbola_derivative(x_vals, y_vals, m):
    """Calculates the derivative of the cost function in terms of m"""
    
    der = 0.0
    for index in range(len(x_vals)):
        x = x_vals[index]
        y = y_vals[index]
        
        if y != 0:
            outcome = hyperbola_prediction(x, m)
            partial = 0.5 * (1 + (m*x)**2)**(-.5) * 2.0*m*x**2 - x
            der += (outcome - y) * partial / len(x_vals)
    
    return der


def hyperbola_integral(m, endpoint):
    """Calculates the integral of the hyperbola given by m, from 0 to the endpoint"""
    
    m = float(m)
    endpoint = float(endpoint)
    integral = (math.asinh(m*endpoint) + m*endpoint*math.sqrt((m*endpoint)**2 + 1))/(2*m)
    
    return integral


def fit_hyperbola(x_vals, y_vals, alpha):
    """Uses gradient descenet to fit a hyperbola to the data"""
    
    m = 1.0
    old_der = hyperbola_derivative(x_vals, y_vals, m)
    der = old_der
    iteration = 0
    cost = 1.0
    
    while old_der * der > 0.0 and abs(der) > 0.001 and cost > 0.001:
        der = hyperbola_derivative(x_vals, y_vals, m)
        m -= der*alpha
        iteration += 1
        cost = hyperbola_cost(x_vals, y_vals, m)
    
    return m


def color_pair_m_prob(color1, color2, m_values):
    """Uses hyperbolic probability distributions to determine the probability that an object of color1 would also have a pixel of color2 (all given in h/s/v)"""
    
    hue_diff = min(abs(color1[0] - color2[0]), 1 - abs(color1[0] - color2[0]))
    sat_diff = abs(color1[1] - color2[1])
    val_diff = abs(color1[1] - color2[1])
    hue_prob = hyperbola_prediction(100*hue_diff, 2*m_values[0]) * (1.0 - hyperbola_integral(2*m_values[0], 100*hue_diff)/hyperbola_integral(2*m_values[0], 20.0))
    sat_prob = hyperbola_prediction(100*sat_diff, m_values[1]) * (1.0 - hyperbola_integral(m_values[1], 100*sat_diff)/hyperbola_integral(m_values[1], 20.0))
    val_prob = hyperbola_prediction(100*val_diff, m_values[2]) * (1.0 - hyperbola_integral(m_values[2], 100*val_diff)/hyperbola_integral(m_values[2], 20.0))
    prob = hue_prob * sat_prob * val_prob
    
    return prob


def euclid_color_dist(color1, color2):
    """Computes the distance squared between two colors given by RGB values, using the standard metric"""
    
    return (float(color1[0] - color2[0]))**2 + (float(color1[1] - color2[1]))**2 + (float(color1[2] - color2[2]))**2


def find_border(image):
    """Returns a list of the border pixels in an image"""
    
    rows = range(len(image))
    columns = range(len(image[0]))
    border = []
    
    for row in rows:
        if row == 0 or row == len(rows)-1:
            for column in columns:
                border.append([row, column])
        else:
            border.append([row, 0])
            border.append([row, len(columns)-1])
    
    return border


def pixel_neighbors(pixel, height, width):
    """Returns a list of the neighbors of a pixel, including corners"""
    
    neighbors = []
    for row in range(max(pixel[0]-1, 0), min(pixel[0]+2, height)):
        for col in range(max(pixel[1]-1, 0), min(pixel[1]+2, width)):
            if [row, col] != pixel:
                neighbors.append([row, col])
    
    return neighbors


def pixel_four_neighbors(pixel, height, width):
    """Returns a list of the neighbors of a pixel, not including corners"""
    
    neighbors = []
    if pixel[0] > 0:
        neighbors.append([pixel[0] - 1, pixel[1]])
    if pixel[1] > 0:
        neighbors.append([pixel[0], pixel[1] - 1])
    if pixel[0] < height - 1:
        neighbors.append([pixel[0] + 1, pixel[1]])
    if pixel[1] < width - 1:
        neighbors.append([pixel[0], pixel[1] + 1])
    
    return neighbors


def forward_pixel_neighbors(pixel, height, width):
    """Returns a list of neighbors to the right and/or below the given pixel"""
    
    neighbors = []
    for row in range(max(pixel[0]-1, 0), min(pixel[0]+2, height)):
        for col in range(max(pixel[1]-1, 0), min(pixel[1]+2, width)):
            if row > pixel[0] or col > pixel[1]:
                neighbors.append([row, col])
    
    return neighbors


def circular_pixel_neighbors(pixel, height, width, radius):
    """Returns a list of all pixels within a certain radius of the given pixel"""
    
    neighbors = []
    for row in range(int(max(pixel[0] - math.ceil(radius), 0)), int(min(pixel[0] + math.ceil(radius) + 1, height))):
        for col in range(int(max(pixel[1] - math.ceil(radius), 0)), int(min(pixel[1] + math.ceil(radius) + 1, width))):
            if [row, col] != pixel and math.sqrt((row - pixel[0])**2 + (col - pixel[1])**2) <= radius:
                neighbors.append([row, col])
    
    return neighbors


def known_gaussian_smoothing(background_prob, min_prob, sigma):
    """Given probabilities that each pixel is part of the background, returns a list of definite background pixels"""
    """Also applies a Gaussian blur to the resulting collection of known pixels"""
    
    known_list = []
    border_pixels = find_border(background_prob)
    height = len(background_prob)
    width = len(background_prob[0])
    known_pixels = np.zeros((height, width))
    
    for row in range(height):
        for col in range(width):
            prob = background_prob[row][col]
            if prob > min_prob or [row, col] in border_pixels:
                known_pixels[row][col] = 1.0
    new_known = np.zeros((height, width))
    
    for row in range(height):
        for col in range(width):
            current_knowledge = 2*known_pixels[row][col] - 1.0
            total_weight = 1.0
            neighbors = circular_pixel_neighbors([row, col], height, width, 2*sigma)
            for neighbor in neighbors:
                new_knowledge = 2*known_pixels[neighbor[0]][neighbor[1]] - 1.0
                weight = gaussian(math.sqrt((row - neighbor[0])**2 + (col - neighbor[1])**2), sigma)
                current_knowledge += new_knowledge*weight
                total_weight += weight
            if current_knowledge > 0.0:
                new_known[row][col] = 1.0
            elif current_knowledge < 0.0:
                new_known[row][col] = 0.0
            else:
                new_known[row][col] = known_pixels[row][col]
    
    for row in range(height):
        for col in range(width):
            if new_known[row][col] == 1.0:
                known_list.append([row, col])
    
    return [known_list, new_known]


def largest_connected_component(known):
    """Determines the largest connected component consisting of known foreground pixels"""
    
    height = len(known)
    width = len(known[0])
    new_known = np.ones((height, width)) - known
    connected_components = []
    
    while np.sum(new_known) > 0.0:
        connected_component = np.zeros((height, width))
        
        """This part finds the first foreground pixel which hasn't yet been categorized."""
        current_pixel = [0,0]
        done = False
        
        while new_known[current_pixel[0]][current_pixel[1]] != 1.0 and done == False:
            if current_pixel[1] < width - 1:
                current_pixel[1] += 1
            elif current_pixel[0] < height - 1:
                current_pixel[0] += 1
                current_pixel[1] = 0
            else:
                done = True
        
        """This part used breadth first search to find the connected component
        containing the current pixel."""
        if new_known[current_pixel[0]][current_pixel[1]] == 1.0:
            boundary = [current_pixel]
            connected_component[current_pixel[0]][current_pixel[1]] = 1.0
            new_known[current_pixel[0]][current_pixel[1]] = 0.0
            
            while len(boundary) > 0:
                current_pixel = boundary.pop(0)
                new_known[current_pixel[0]][current_pixel[1]] = 0.0
                neighbors = pixel_neighbors(current_pixel, height, width)
                
                for neighbor in neighbors:
                    if new_known[neighbor[0]][neighbor[1]] == 1.0:
                        boundary.append(neighbor)
                        connected_component[neighbor[0]][neighbor[1]] = 1.0
                        new_known[neighbor[0]][neighbor[1]] = 0.0
        
        if np.sum(connected_component) > 0.0:
            connected_components.append(connected_component)
    
    """Now that we have a list of connected components, we find the largest one."""
    best_size = 0.0
    biggest_component = new_known
    
    for component in connected_components:
        size = np.sum(component)
        if size > best_size:
            best_size = size
            biggest_component = component
    biggest_component = np.ones((height, width)) - biggest_component
    
    return biggest_component


def find_min_prob(image):
    """Determines a probability cutoff for classifying foreground pixels, based on the image's size and color variation"""
    
    height = len(image)
    width = len(image[0])
    area = float(height * width)
    image_sigmas = sigmas(image)
    sigmas_prod = image_sigmas[0] * image_sigmas[1] * image_sigmas[2]
    prob = math.log(area / 10000.0 + 1)/(12260.0 * sigmas_prod)
    
    return prob


def find_foreground(image, m_values, known, min_prob):
    """Uses a breadth first search process to find the foreground of an image."""
    height = len(image)
    width = len(image[0])
    image_hsv = skimage.color.convert_colorspace(image, 'RGB', 'HSV')
    old_boundary = known
    new_boundary = []
    background_prob = np.zeros((height, width))
    knowledge = np.zeros((height, width))
    
    for pixel in old_boundary:
        background_prob[pixel[0]][pixel[1]] = 1.0
        knowledge[pixel[0]][pixel[1]] = .7
    
    while len(old_boundary) > 0:
        for current_pixel in old_boundary:
            knowledge[current_pixel[0]][current_pixel[1]] = 1.0
            current_color = image_hsv[current_pixel[0]][current_pixel[1]]
            neighbors = pixel_four_neighbors(current_pixel, height, width)
            
            for neighbor in neighbors:
                if knowledge[neighbor[0]][neighbor[1]] == 0.0:
                    new_boundary.append(neighbor)
                    knowledge[neighbor[0]][neighbor[1]] = 0.3
                if knowledge[neighbor[0]][neighbor[1]] == 0.3:
                    new_color = image_hsv[neighbor[0]][neighbor[1]]
                    new_prob = color_pair_m_prob(current_color, new_color, m_values)
                    old_prob = background_prob[neighbor[0]][neighbor[1]]
                    if new_prob > old_prob:
                        background_prob[neighbor[0]][neighbor[1]] = new_prob
        
        old_boundary = []
        
        for pixel in new_boundary:
            if background_prob[pixel[0]][pixel[1]] > min_prob:
                old_boundary.append(pixel)
        
        new_boundary = []
    
    return [background_prob, knowledge]


def layered_foreground(image_list, min_prob_2):
    """Uses a list of rescaled versions of an image to find the foreground of the original image. 
    Uses k-means clustering to determine the appropriate Gaussian for colors in each cluster."""
    
    known = find_border(image_list[-1])
    known_image = image_list[-1]
    current_image = image_list[-1]
    
    for index in range(len(image_list)):
        current_index = len(image_list) - 1 - index
        current_image = image_list[current_index]
        m_values = find_m_values(current_image)
        height = len(current_image)
        width = len(current_image[0])
        min_prob_1 = find_min_prob(current_image)
        probs = find_foreground(current_image, m_values, known, min_prob_1)
        foreground = probs[1]
        
        if current_index > 0:
            height = len(image_list[current_index - 1])
            width = len(image_list[current_index - 1][0])
        else:
            height = len(image_list[current_index])
            width = len(image_list[current_index][0])
        
        bigger_foreground = resize(foreground, (height, width))
        smaller_dim = min(height, width)
        known_outcome = known_gaussian_smoothing(bigger_foreground, min_prob_2, float(smaller_dim)/100.0)
        known = known_outcome[0]
        known_image = known_outcome[1]
    
    return known_image


"""Loads an image and detects its foreground"""
image_to_parse = io.imread("Latte.jpg")

scaled_image_list = create_scaled_images(image_to_parse)

foreground = layered_foreground(scaled_image_list, 0.9)

connected = largest_connected_component(foreground)

plt.imshow(connected)
plt.show()
