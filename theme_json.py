import json
import os
import sys
import numpy as np

def generate_json(name, version, color_frame, color_toolbar, color_ntp_next, color_ntp_text,
                  color_ntp_link, color_ntp_section, color_button_background, hsl_tints, 
                  ntp_background_alignment = "bottom", ntp_background_repeat = "no-repeat"):

    json_obj = {
                   "name" : name, 
                   "version" : version,
                   "theme" : {
                       "images" : {
                           "theme_frame" : "images/theme_frame.png",
                           "theme_frame_overlay" : "images/theme_frame_overlay.png",
                           "theme_toolbar" : "images/theme_toolbar.png",
                           "theme_ntp_background" : "images/theme_ntp_background.png",
                           "theme_ntp_attribution" : "images/theme_ntp_attribution.png"
                       },
                        "colors" : {
                            "frame" : color_frame,
                            "toolbar" : color_toolbar,
                            "ntp_text" : color_ntp_next,
                            "ntp_link" : color_ntp_link,
                            "ntp_section" : color_ntp_section,
                            "button_background" : color_button_background
                        },
                        "tints" : {
                            "buttons" : hsl_tints
                        },
                        "properties" : {
                            "ntp_background_alignment" : ntp_background_alignment,
                            "ntp_background_repeat" : ntp_background_repeat 
                        }
                   }, 
               }    
    
    return json_obj

if len(sys.argv) != 2:
    print("Length of sys.argv is not correct.")
    exit(0)
name = sys.argv[1]
material_path = "material" + os.sep + name + os.sep + name + ".npy"
colors = np.load(material_path).tolist()
print(colors)
base_dir = "chrome_theme" + os.sep + name + os.sep
version = "1.0"
color_frame = colors[0]
color_toolbar = colors[1]
color_ntp_next = colors[2]
color_ntp_text = colors[3]
color_ntp_link = colors[4]
color_ntp_section = colors[5]
color_button_background = colors[6]
hsl_tints = colors[7]
ntp_background_alignment = "bottom"
ntp_background_repeat = "no-repeat"

if not os.path.exists(base_dir):
    os.mkdir(base_dir)
#
if not os.path.exists(base_dir + "images"):
    os.mkdir(base_dir + "images")

json_obj = generate_json(name, version, color_frame, color_toolbar, color_ntp_next, color_ntp_text,
                  color_ntp_link, color_ntp_section, color_button_background, hsl_tints, 
                  ntp_background_alignment, ntp_background_repeat)

json_str = json.dumps(json_obj)
print(json_str)

json_file_path = base_dir + "manifest.json"
json_file = open(json_file_path, 'w')
json_file.write(json_str)
json_file.close()
