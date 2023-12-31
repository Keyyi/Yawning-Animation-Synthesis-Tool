import maya.cmds as cmds
import pickle
import pandas as pd
import numpy as np
import random
from scipy.stats import norm
from scipy.special import gamma
from scipy.optimize import minimize_scalar
import pymel.core as pm
from perlin_noise import PerlinNoise
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import json

created_object_list = []

def gennorm_pdf(x, beta, mean, sigma):
    return np.exp(-np.abs((x - mean) / sigma)**beta) / (2 * sigma * gamma(1/beta) * (1/beta)**(1/beta))

# Define a skewing function using the CDF of the normal distribution
def skewing_function(x, alpha):
    return 2 * norm.cdf(alpha * x)

# Internal function to calculate the amplitude A
def _calculate_amplitude(beta, max_value, sigma, y_min, mean, alpha):
    def objective_function(A):
        y_values = A * gennorm_pdf(x_values, beta, mean, sigma) * skewing_function(x_values, alpha) + y_min
        return np.abs(max(y_values) - max_value)

    x_values = np.linspace(mean - 5*sigma, mean + 5*sigma, 1000)
    result = minimize_scalar(objective_function)
    return result.x

# Modified function to calculate the skewed GND PDF
def skewed_gennorm_pdf(x, beta, max_value, sigma, y_min, mean, alpha=5):
    A = _calculate_amplitude(beta, max_value, sigma, y_min, mean, alpha)
    return A * gennorm_pdf(x, beta, mean, sigma) * skewing_function(x, alpha) + y_min

# Generate perlin noise
def generate_perlin_noise(x, y):
    dy = np.gradient(y, x)
    threshold = 0.1
    smooth_regions = np.abs(dy) < threshold
    noise = PerlinNoise(octaves=0.3)
    scale = 0.1
    y_noise = np.array([noise([xi * scale]) if smooth else 0 for xi, smooth in zip(x, smooth_regions)])
    
    return y_noise

######################## Helper Functions to Generate Random Blink ########################
def is_too_close(new_x, selected_x, min_distance):
    for point in selected_x:
        if abs(new_x - point) < min_distance:
            return True
    return False

def segment_and_normalize_pdf(x, y_pdf, x_range):
    indices = (x >= x_range[0]) & (x <= x_range[1])
    x_segment = x[indices]
    y_pdf_segment = y_pdf[indices]
    area = simps(y_pdf_segment, x_segment)
    y_pdf_segment_normalized = y_pdf_segment / area
    return x_segment, y_pdf_segment_normalized

def random_points(x, y, n=1):
    cdf = cumtrapz(y, x, initial=0)
    cdf /= cdf[-1]  # Normalize
    inverse_cdf = interp1d(cdf, x)
    random_samples = np.random.rand(n)
    return inverse_cdf(random_samples)
    
def random_peaks(x_smooth, y_smooth_pdf, num_points):
    length = 1 / num_points
    min_distance = length / 2
    ranges = [(i * length, (i + 1) * length) for i in range(num_points)]

    selected_x = []

    for x_range in ranges:
        x_segment, y_pdf_segment = segment_and_normalize_pdf(x_smooth, y_smooth_pdf, x_range)

        attempts = 0
        while attempts < 10:
            random_x = random_points(x_segment, y_pdf_segment)[0]
            if not is_too_close(random_x, selected_x, min_distance):
                selected_x.append(random_x)
                break
            attempts += 1
    return selected_x
#################################### End of Helper Functions ####################################

####################### Helper Functions to Generate Eye Blendshape Curves ######################
def find_second_peak_index(curve, mean):
    argmax = np.argmax(curve[mean:])
    return mean + argmax

# Find the first intersection point between two curves after xt. Returns the x-value of the first intersection point.
def find_first_intersection(x, curve1, curve2, xt):
    for i in range(xt, len(x) - 1):
        if (curve1[i] - curve2[i]) * (curve1[i+1] - curve2[i+1]) <= 0:
            return x[i]
    return None
#################################### End of Helper Functions ####################################

def generate_yawning(*args):
    openness = cmds.floatSliderGrp(openness_value, query=True, value=True)
    duration = int(cmds.floatSliderGrp(duration_value, query=True, value=True) * 60)
    mouth_crook = cmds.floatSliderGrp(crookness_value, query=True, value=True)
    eyes_open = cmds.checkBox(eyes_open_checkbox, query=True, value=True)

    with open('/Users/keyizhang/Downloads/models.pickle', 'rb') as handle:
        models = pickle.load(handle)

    with open('/Users/keyizhang/Downloads/eye_data.json', 'r') as file:
        blink_data = json.load(file)

    jaw_open_max_value = openness
    blendshapes = ["MouthStretchLeft", "MouthStretchRight", "MouthLowerDownLeft",
                "MouthLowerDownRight", "MouthUpperUpLeft", "MouthUpperUpRight",
                "JawForward", "MouthFunnel", 'MouthShrugLower', 'MouthShrugUpper',
                "MouthClose", "MouthPucker", "MouthRollLower", "MouthRollUpper",
                "MouthPressLeft", "MouthPressRight", "CheekPuff", "CheekSquintLeft",
                "CheekSquintRight", "NoseSneerLeft", "MouthSmileLeft", "MouthDimpleLeft",
                "HeadPitch"]
    curves = {}
    time_data = pd.RangeIndex(start=0, stop=duration, step=1)
    jaw_open_beta = models["JawOpen_beta"].predict([1, jaw_open_max_value])[0]
    jaw_open_mean = models["JawOpen_mean"].predict([1, duration])[0]
    jaw_open_sigma = models["JawOpen_sigma"].predict([1, duration])[0]
    jaw_open_y_min = models["JawOpen_y_min"].predict([1, duration, jaw_open_max_value])[0]

    curves["JawOpen"] = skewed_gennorm_pdf(time_data, jaw_open_beta, jaw_open_max_value, jaw_open_sigma, jaw_open_y_min, jaw_open_mean)

    for blendshape in blendshapes:
        if blendshape == "HeadPitch":
            max_value = (models[f"NoseSneerLeft_max_value"].predict([1, jaw_open_max_value])[0])/4
            beta = models[f"NoseSneerLeft_beta"].predict([1, jaw_open_beta])[0]
            mean = models[f"NoseSneerLeft_mean"].predict([1, jaw_open_mean])[0]
            sigma = models[f"NoseSneerLeft_sigma"].predict([1, jaw_open_sigma])[0]
            y_min = max(models[f"NoseSneerLeft_y_min"].predict([1, jaw_open_y_min])[0], 0)            
        if blendshape != "MouthSmileLeft" and blendshape != "MouthDimpleLeft" and blendshape != "HeadPitch":
            max_value = models[f"{blendshape}_max_value"].predict([1, jaw_open_max_value])[0]
            beta = models[f"{blendshape}_beta"].predict([1, jaw_open_beta])[0]
            mean = models[f"{blendshape}_mean"].predict([1, jaw_open_mean])[0]
            sigma = models[f"{blendshape}_sigma"].predict([1, jaw_open_sigma])[0]
            y_min = max(models[f"{blendshape}_y_min"].predict([1, jaw_open_y_min])[0], 0)
        if blendshape == "MouthShrugLower" or blendshape == "MouthShrugUpper" or \
            blendshape == "MouthRollLower" or blendshape == "MouthRollUpper" or \
            blendshape == "MouthPressLeft" or blendshape == "MouthPressRight" or \
            blendshape == "CheekPuff" or blendshape == "CheekSquintLeft" or \
            blendshape == "CheekSquintRight":
            fitted_curve = 1-skewed_gennorm_pdf(time_data, beta, max_value, sigma, y_min, mean)
        elif blendshape == "MouthSmileLeft":
            fitted_curve = skewed_gennorm_pdf(time_data, jaw_open_beta, jaw_open_max_value/2, jaw_open_sigma, 0, duration+20)
        elif blendshape == "MouthDimpleLeft":
            fitted_curve = skewed_gennorm_pdf(time_data, jaw_open_beta, jaw_open_max_value/4, jaw_open_sigma, 0, duration+20)
        else:
            fitted_curve = skewed_gennorm_pdf(time_data, beta, max_value, sigma, y_min, mean)
        if fitted_curve.min() < 0:
            fitted_curve = 0
        curves[blendshape] = fitted_curve

    all_columns = ['Timecode', 'EyeBlinkLeft', 'EyeLookDownLeft',
       'EyeLookInLeft', 'EyeLookOutLeft', 'EyeLookUpLeft', 'EyeSquintLeft',
       'EyeWideLeft', 'EyeBlinkRight', 'EyeLookDownRight', 'EyeLookInRight',
       'EyeLookOutRight', 'EyeLookUpRight', 'EyeSquintRight', 'EyeWideRight',
       'JawForward', 'JawRight', 'JawLeft', 'JawOpen', 'MouthClose',
       'MouthFunnel', 'MouthPucker', 'MouthRight', 'MouthLeft',
       'MouthSmileLeft', 'MouthSmileRight', 'MouthFrownLeft',
       'MouthFrownRight', 'MouthDimpleLeft', 'MouthDimpleRight',
       'MouthStretchLeft', 'MouthStretchRight', 'MouthRollLower',
       'MouthRollUpper', 'MouthShrugLower', 'MouthShrugUpper',
       'MouthPressLeft', 'MouthPressRight', 'MouthLowerDownLeft',
       'MouthLowerDownRight', 'MouthUpperUpLeft', 'MouthUpperUpRight',
       'BrowDownLeft', 'BrowDownRight', 'BrowInnerUp', 'BrowOuterUpLeft',
       'BrowOuterUpRight', 'CheekPuff', 'CheekSquintLeft', 'CheekSquintRight',
       'NoseSneerLeft', 'NoseSneerRight', 'TongueOut', 'HeadYaw', 'HeadPitch',
       'HeadRoll', 'LeftEyeYaw', 'LeftEyePitch', 'LeftEyeRoll', 'RightEyeYaw',
       'RightEyePitch', 'RightEyeRoll']
    yawning = pd.DataFrame(np.zeros((duration, len(all_columns))), columns=all_columns)
    yawning['Timecode'] = [i/60 for i in range(duration)]
    blendshapes = ["JawOpen"] + blendshapes
    for blendshape in blendshapes:
        yawning[blendshape] = curves[blendshape]
    yawning["NoseSneerRight"] = curves["NoseSneerLeft"]
    yawning["MouthSmileRight"] = curves["MouthSmileLeft"]
    yawning["MouthDimpleRight"] = curves["MouthDimpleLeft"]

    x = np.linspace(0, duration+1, duration)
    if eyes_open:
        num_points = round(duration/blink_data['avg_time'])

        # Randomly generating blink peaks and duration
        selected_peak = random_peaks(np.asarray(blink_data['peak']['x_smooth']), np.asarray(blink_data['peak']['y_smooth_pdf']), num_points)
        selected_peak = np.sort((np.array(selected_peak) * duration).astype(int))

        selected_duration = (random_points(np.asarray(blink_data['duration']['x_smooth']), np.asarray(blink_data['duration']['y_smooth_pdf']), num_points)).astype(int)
        if min(selected_duration) >= 20:
            idx = np.argmax(selected_duration)
            selected_duration[idx] = random.randint(8, 19)

        eye_blink = np.zeros(x.shape)
        eye_lookdown = np.zeros(x.shape)
        alpha = 20
        prev_mean = 0
        inter_x = 0
        inter_x_look = 0
        for i in range(num_points):
            if i == 0:
                y_min = random.uniform(0.05, 0.15)
            elif i != (num_points - 1):
                y_min = random.uniform(0.2, 0.3)
            mean = selected_peak[i]
            max_value = random.uniform(0.8, 1)
            sigma = selected_duration[i]/2
            beta = selected_duration[i]/5.65
            cur_blink = skewed_gennorm_pdf(x, beta, max_value, sigma, y_min, mean, alpha)
            tmp = skewed_gennorm_pdf(x, beta, max_value - y_min, sigma, y_min, mean, alpha)
            cur_lookdown = np.minimum(cur_blink, 1 - tmp)

            if eye_blink.all() == 0:
                eye_blink = cur_blink
                eye_lookdown = cur_lookdown
            else:
                inter_x = find_first_intersection(x, eye_blink, cur_blink, prev_mean)
                inter_x_look = find_first_intersection(x, eye_lookdown, cur_lookdown, second_peak_idx)
                if inter_x is not None:
                    idx = np.where(x >= inter_x)[0][0]
                    eye_blink[idx:] = cur_blink[idx:]
                if inter_x_look is not None:
                    idx_look = np.where(x >= inter_x_look)[0][0]
                    eye_lookdown[idx_look:] = cur_lookdown[idx_look:]
            prev_mean = mean
            second_peak_idx = find_second_peak_index(cur_lookdown, mean)

        # Smoothly decrease the last part of the curve
        b = duration
        a = second_peak_idx
        decrease_factors = np.linspace(eye_blink[a], 0.05, b - a)
        eye_blink[a:b] = eye_blink[a:b] * decrease_factors
        eye_lookdown[a:b] = eye_lookdown[a:b] * decrease_factors

        # Add noise to make eye blinks more natural
        noise1 = generate_perlin_noise(x, eye_blink)
        noise2 = generate_perlin_noise(x, eye_lookdown)
        eye_blink_noisy = eye_blink + 0.1 * noise1
        eye_lookdown_noisy = eye_lookdown * 0.98 + 0.1 * noise2 - 0.02
        squint = 0.1 + 0.1 * eye_lookdown_noisy
        half = (min(eye_lookdown) - min(squint))/2
        pitch = (eye_lookdown_noisy - min(eye_lookdown))/2 + min(eye_lookdown) - half
        brow = skewed_gennorm_pdf(x, jaw_open_beta, jaw_open_max_value/2, jaw_open_sigma - 10, jaw_open_y_min, jaw_open_mean+10)

    if not eyes_open:
        beta = 50
        max_value = random.uniform(0.95, 1)
        sigma = jaw_open_sigma + 10
        y_min = random.uniform(0.05, 0.1)
        mean = jaw_open_mean
        alpha = 20
        
        eye_blink = skewed_gennorm_pdf(x, beta, max_value, sigma, y_min, mean, alpha)
        tmp = skewed_gennorm_pdf(x, beta, max_value - y_min, sigma, y_min, mean, alpha)
        eye_lookdown = np.minimum(eye_blink, 1 - tmp)
        frames = jaw_open_mean - jaw_open_sigma

        # Randomly add one blink before or after the eye-closing process
        if frames > 20:
            combined_range = list(range(5, int(frames))) + list(range(int(jaw_open_mean + jaw_open_sigma), duration - 5))
            selected_peak = random.choice(combined_range)
            selected_duration = random.randint(8, 11)
            
            mean = selected_peak
            max_value = random.uniform(0.8, 1)
            sigma = selected_duration/2
            beta = selected_duration/5.65

            eye_blink_rand = skewed_gennorm_pdf(x, beta, max_value, sigma, y_min, mean, alpha)
            eye_blink = np.maximum(eye_blink, eye_blink_rand)
            tmp = skewed_gennorm_pdf(x, beta, max_value, sigma, y_min, mean, alpha)
            eye_lookdown_rand = np.minimum(eye_blink_rand, 1 - tmp)
            eye_lookdown = np.maximum(eye_lookdown, eye_lookdown_rand)

        noise1 = generate_perlin_noise(x, eye_blink)
        noise2 = generate_perlin_noise(x, eye_lookdown)
        eye_blink_noisy = eye_blink + 0.1 * noise1
        eye_lookdown_noisy = eye_lookdown * 0.98 + 0.1 * noise2 - 0.02
        squint = 0.1 + 0.1 * eye_lookdown_noisy
        half = (min(eye_lookdown) - min(squint))/2
        pitch = (eye_lookdown_noisy - min(eye_lookdown))/2 + min(eye_lookdown) - half
        brow = skewed_gennorm_pdf(x, jaw_open_beta, jaw_open_max_value/4*3, jaw_open_sigma - 10, jaw_open_y_min, jaw_open_mean+10)
    
    yawning["BrowDownLeft"] = brow
    yawning["BrowDownRight"] = brow
    yawning["EyeBlinkLeft"] = eye_blink_noisy
    yawning["EyeBlinkRight"] = eye_blink_noisy
    yawning["EyeSquintLeft"] = squint
    yawning["EyeSquintRight"] = squint
    yawning['EyeLookDownLeft'] = eye_lookdown_noisy
    yawning['EyeLookDownRight'] = eye_lookdown_noisy
    yawning['LeftEyePitch'] = pitch
    yawning['RightEyePitch'] = pitch


    target_jaw_open = models["JawRight"].predict([1, jaw_open_max_value])[0]
    tolerance = 5e-2
    matching_rows = yawning[(yawning['JawOpen'] >= target_jaw_open - tolerance) & (yawning['JawOpen'] <= target_jaw_open + tolerance) & (yawning.index > jaw_open_mean)]
    latest_timecode = matching_rows['Timecode'].max()
    if yawning[yawning['Timecode'] == latest_timecode].empty:
        fitted_curve = 0
    else:
        jaw_right_mean = yawning[yawning['Timecode'] == latest_timecode].index[0]
        jaw_right_max_ratio = 0.35
        fitted_curve = skewed_gennorm_pdf(time_data, 2, jaw_right_max_ratio*jaw_open_max_value*mouth_crook, 10, 0, jaw_right_mean)
    if not np.isscalar(fitted_curve) and fitted_curve.min() < 0:
        fitted_curve = 0
    yawning["JawRight"] = fitted_curve
    yawning["MouthRight"] = fitted_curve


    columns = [column for column in all_columns if column not in ["Timecode"]]
    name_to_mesh = {"eyesquintright": "Mesh31", "eyelookupright": "Mesh33", "jawright": "Mesh25", "mouthstretchright": "Mesh4", "browdownleft": "Mesh50", "cheekpuff": "Mesh45", "mouthsmileleft": "Mesh5", "jawopen": "Mesh26", "mouthshrugupper": "Mesh8", "mouthleft": "Mesh18", "mouthsmileright": "Mesh7", "jawforward": "Mesh28", "mouthrollupper": "Mesh10", "mouthlowerdownright": "Mesh16", "browouterupright": "Mesh46", "mouthdimpleright": "Mesh22", "eyelookdownleft": "Mesh40", "eyewideleft": "Mesh30", "mouthshruglower": "Mesh9", "mouthpressright": "Mesh14", "mouthrolllower": "Mesh11", "jawleft": "Mesh27", "cheeksquintright": "Mesh43", "mouthpucker": "Mesh13", "mouthdimpleleft": "Mesh23", "mouthupperupleft": "Mesh2", "eyelookdownright": "Mesh39", "nosesneerright": "Mesh", "eyelookoutleft": "Mesh36", "eyewideright": "Mesh29", "browinnerup": "Mesh48", "cheeksquintleft": "Mesh44", "eyelookupleft": "Mesh34", "mouthlowerdownleft": "Mesh17", "browouterupleft": "Mesh47", "eyelookinleft": "Mesh38", "eyelookinright": "Mesh37", "mouthright": "Mesh12", "mouthstretchleft": "Mesh6", "mouthpressleft": "Mesh15", "nosesneerleft": "Mesh1", "mouthupperupright": "Mesh3", "mouthclose": "Mesh24", "eyelookoutright": "Mesh35", "browdownright": "Mesh49", "eyeblinkleft": "Mesh42", "mouthfrownleft": "Mesh21", "mouthfrownright": "Mesh20", "eyeblinkright": "Mesh41", "mouthfunnel": "Mesh19", "eyesquintleft": "Mesh32"}
    times = yawning['Timecode'].tolist()
    yawning = yawning.drop(['Timecode'], axis=1)
    cal_data = yawning.values.tolist()

    for j in range(0, len(columns)-10):
        name = columns[j].lower()
        try:
            weight_name = name_to_mesh[name]
        except:
            if name[-2:] == "_l":
                name = name[:-2] + "left"
            else:
                name = name[:-2] + "right"
            weight_name = name_to_mesh[name]
        cmds.cutKey("blendShape1.{}".format(weight_name), s=True)
    cmds.cutKey("Neutral:Mesh.{}".format("rotateY"))
    cmds.cutKey("Neutral:Mesh.{}".format("rotateX"))
    cmds.cutKey("Neutral:Mesh.{}".format("rotateZ"))
    # compute curve using loaded data:
    for i in range(0, len(times)):
        for j in range(0, len(columns)-10):
            name = columns[j].lower()
            try:
                weight_name = name_to_mesh[name]
            except:
                if name[-2:] == "_l":
                    name = name[:-2] + "left"
                else:
                    name = name[:-2] + "right"
                weight_name = name_to_mesh[name]
            cmds.setKeyframe("blendShape1.{}".format(weight_name), v=float(cal_data[i][j]),
                                 t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateY"), v=(float(cal_data[i][-9]))*-90,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateX"), v=(float(cal_data[i][-8]))*-70,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
        cmds.setKeyframe("Neutral:Mesh.{}".format("rotateZ"), v=(float(cal_data[i][-7]))*-45,
                     t=times[i] * mel.eval('float $fps = `currentTimeUnitToFPS`'))
    
    sp, _ = cmds.sphere(r = 10, pivot =[0, 0, 0])
    cmds.select(sp)
    cmds.move(0, 150, 100)
    created_object_list.append(sp)
    cmds.addAttr(sp, ln='object_type', at="float", k=True)
    cmds.setAttr(sp+".object_type", 1)
    cmds.addAttr(sp, ln='object_interestingness', at="float", k=True)
    cmds.setAttr(sp+".object_interestingness", 0.05)

# Define the function for creating the UI window
def create_ui():
    window_name = "MyWindow"
    if cmds.window(window_name, exists=True):
        cmds.deleteUI(window_name)
    window = cmds.window(window_name, title="Yawning Generator", widthHeight=(1000, 200))

    # Create two buttons
    # cmds.rowColumnLayout(numberOfColumns=2, columnWidth=[(1, 200), (2, 200)])
    cmds.rowColumnLayout(numberOfColumns=1, columnWidth=[1, 500])
    global openness_value
    openness_value = cmds.floatSliderGrp(label='Mouth Open', field=True, minValue=0.0, maxValue=1.0, value=0.5, columnWidth=[(1, 300), (2, 100), (3, 100)])

    global duration_value
    duration_value = cmds.floatSliderGrp(label='Duration in seconds', field=True, minValue=3, maxValue=7, value=5, columnWidth=[(1, 300), (2, 100), (3, 100)])

    global crookness_value
    crookness_value = cmds.floatSliderGrp(label='Mouth Crooked', field=True, minValue=0.0, maxValue=1.0, value=0.0, columnWidth=[(1, 300), (2, 100), (3, 100)])

    global eyes_open_checkbox
    eyes_open_checkbox = cmds.checkBox(label='Eyes Open', value=False)

    cmds.button(label="Generate", command=generate_yawning)
    cmds.showWindow(window)

# Call the function to create the UI window
create_ui()