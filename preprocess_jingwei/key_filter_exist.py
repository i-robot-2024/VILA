import json
import os
fp_write = open("/mnt/share_disk/jintao/VILA/playground/data/sharegpt_video/video_caption_pretrain_exist.json",'w')
base_dir = "/mnt/share_disk/jintao/VILA/playground/data/sharegpt_video/videos/"
with open("/mnt/share_disk/jintao/VILA/playground/data/sharegpt_video/video_caption_pretrain.json", "r") as fp:   
                list_data_dict = []
                for q in fp:
                    line = json.loads(q) 
                    if os.path.exists(base_dir+line['id']):
                            list_data_dict.append(q)
                fp_write.writelines(list_data_dict)
fp_write.close()