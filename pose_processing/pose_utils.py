import json
import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import matplotlib.animation as animation
import imageio
from scipy.ndimage import gaussian_filter
from scipy.stats import multivariate_normal
import cv2

FACE_INDECIES = list(range(27)) # outer shape of the face 0 - 26
FACE_JOINT_NAMES_X = [f'x_{a}' for a in FACE_INDECIES]
FACE_JOINT_NAMES_Y = [f'y_{a}' for a in FACE_INDECIES]

POSE_FOLDER_NAMES_MAP = {'face': 'openface', 'body_hand': 'openpose'}
POSE_EXTENTION_MAP = {'face': 'csv', 'body_hand': 'json'}
POSE_NUMER_JOINTS_MAP = {'face': 27, 'body_hand': 39}

GROUPED_JOINT_MAP = {'face': 0, 'body': 1, 'hand': 2}
heatmap_method_options = ['blur_then_group', 'group_then_blur',None]

VERBOSE = False

class HeatmapGenerator:
    def __init__(self, pose_data, n_frame, n_joints, width, height, sigma=1, heatmap_method='group_then_blur'):
        self.n_frame = n_frame
        self.n_joints = n_joints if heatmap_method is None else 3
        self.width = width
        self.height = height
        self.sigma = sigma
        self.pose_data = pose_data
        self.heatmap = None
        self.grid = None
        self.heatmap_method = heatmap_method
        # self.generate_grid()
        self.init_heatmap()
        self.place_points(pose_data)
        self.apply_gaussian_blur()
        # self.fill_missing_data()
        self.normalize()
    def generate_grid(self):
        F = self.n_frame  # Number of frames
        W = self.width  # Width of the grid
        H = self.height  # Height of the grid
        C = self.n_joints   # Number of channels

        # Create meshgrid for each dimension (x, y)
        grid_x, grid_y = np.mgrid[0:W:1, 0:H:1]

        # # Repeat meshgrid for each channel
        # grid_x = np.repeat(grid_x[:, :, np.newaxis], C, axis=2)
        # grid_y = np.repeat(grid_y[:, :, np.newaxis], C, axis=2)

        # Stack grids along the third dimension to create (W, H, C, 2) grid
        grid = np.dstack((grid_x, grid_y))

        # Repeat grid for each frame
        grid = np.repeat(grid[np.newaxis, ...], C, axis=0)
        grid = np.repeat(grid[np.newaxis, ...], F, axis=0)
        self.grid = grid
    def init_heatmap(self):        
        self.heatmap = np.zeros((self.n_frame, self.n_joints, self.height, self.width))
        
    def place_points(self, pose_data):
        use_conf_as_sigma = False # maybe implement this sometime...
        heat_maps = self.heatmap
        for frame, frame_data in pose_data.items():
            for pose_type, pose_type_data in frame_data.items():
                for person_id, person_data in pose_type_data.items():
                    for joint_id, joint_data in person_data.items():
                        if self.heatmap_method == 'group_then_blur':
                            joint_id = GROUPED_JOINT_MAP[pose_type]
                        
                        # x, y, conf = joint_data['x'], joint_data['y'], joint_data['conf']
                        x, y = joint_data
                        # Check if x, y, and frame are numeric values or NumPy arrays
                        # Check x
                        if not isinstance(x, (int, float, np.integer, np.floating)) or np.isnan(x):
                            continue

                        # Check y
                        if not isinstance(y, (int, float, np.integer, np.floating)) or np.isnan(y):
                            continue

                        # Check frame
                        if not isinstance(frame, (int, float, np.integer, np.floating)) or np.isnan(frame):
                            continue
                        
                        x = np.clip(x, 0, self.width - 1)
                        y = np.clip(y, 0, self.height - 1)                        
                        
                        if frame < 0 or frame >= self.n_frame:
                            continue
                        heat_maps[int(frame), int(joint_id), int(y), int(x)] = 1

        self.heatmap = heat_maps
    def apply_gaussian_blur(self):
        for frame_idx in range(self.n_frame):
            for joint_idx in range(self.n_joints):
                self.heatmap[frame_idx, joint_idx] = gaussian_filter(self.heatmap[frame_idx, joint_idx], sigma=self.sigma)

    def normalize(self):
        max_values = np.amax(self.heatmap, axis=(2, 3))
        for iframe in range(self.n_frame):
            for ijoint in range(self.n_joints):
                if max_values[iframe, ijoint] >0:
                    self.heatmap[iframe, ijoint] /= max_values[iframe, ijoint]
                
    # def fill_missing_data(self):
    #     for joint_idx in range(self.n_joints):
    #         for frame_idx in range(1, self.n_frame - 1):
    #             if np.all(self.heatmap[frame_idx, joint_idx] == 0):
    #                 self.heatmap[frame_idx, joint_idx] = (self.heatmap[frame_idx - 1, joint_idx] + self.heatmap[frame_idx + 1, joint_idx]) / 2


class PoseBaseUtils:
    def __init__(self,
                 pose_data,
                 num_joints=None,
                 image_width=1624,
                 image_height=1224,
                 heat_map_sigma=5,
                 use_conf_as_sigma=False,
                 heatmap_method='group_then_blur',
                 create_heatmap=False,
                 **kwargs):

        self.use_conf_as_sigma = use_conf_as_sigma

        self.num_joints = num_joints
        self.video_path = self.get_path_to_video()
        if image_height is None or image_width is None:
            image_width, image_height = self.get_image_size()

        self.image_width = image_width
        self.image_height = image_height
        self.heat_map_sigma = heat_map_sigma
        self.pose_data = pose_data
        self.marker_color = None
        self.marker_type = 'o'
        self.n_frame = len(pose_data)
        if create_heatmap:
            self.heatmap_generator = HeatmapGenerator(  pose_data,
                                                        self.n_frame,
                                                        num_joints,
                                                        image_width,
                                                        image_height,
                                                        heat_map_sigma,
                                                        heatmap_method, **kwargs)
        else:
            self.heatmap_generator = None
        self.use_conf_as_sigma = use_conf_as_sigma
        

        self.num_joints = num_joints
        self.image_width = image_width
        self.image_height = image_height
        self.heat_map_sigma = heat_map_sigma      
        self.pose_data = pose_data
        self.marker_color = None
        self.marker_type = 'o'
        self.n_frame = len(pose_data)
        self.heatmap_generator = HeatmapGenerator(  pose_data,
                                                    self.n_frame,
                                                    num_joints,
                                                    image_width,
                                                    image_height,
                                                    heat_map_sigma,
                                                    heatmap_method)
    def save_video(self,output_file=None, fps=30):
        if not self.heatmap_generator is None:
            heatmap = self.heatmap_generator.heatmap
        else:
            return
        assert heatmap.shape[1] == 3, f'The number of joint must be 3 to save the video'
        if output_file is None:
            output_file = self.video_path.replace('clips','heatmap')
        if osp.dirname(output_file):
            os.makedirs(osp.dirname(output_file), exist_ok=True)
        save_as_video(heatmap, output_file, fps=30)

    def get_image_size(self, video_path=None):
        if video_path is None:
            video_path = self.video_path
        return get_video_dimensions(video_path)
    
    def get_path_to_video(self, feature_name='clips'):
        
        pose_name = self.pose_name
        root_folder_name = self.root_folder_name
        dataset = self.dataset
        return osp.join(root_folder_name,dataset, pose_name, feature_name,pose_name+'.mp4')
    def get_heat_map(self):
        return self.heatmap_generator.heatmap

    def _draw_skeleton(self, ax=None, save_path=None, scale=1.0):
        if ax is None:
            fig, ax = plt.subplots()

        def update(frame):
            ax.clear()
            ax.set_xlim(0, self.image_width * scale)
            ax.set_ylim(0, self.image_height * scale)
            ax.invert_yaxis()  # Invert y-axis to match image coordinates

            # # Plot body poses
            # for pose_type in self.pose_type_list:
            #     if pose_type in self.pose_data[frame]:
            #         for person in self.pose_data[frame][pose_type].values():
            #             ax.plot(*zip(*person.values()), marker=self.marker_type, linestyle='', markersize=5, color=self.marker_color)
            if self.heatmap_generator is not None:
                heatmap_frame = self.heatmap_generator.heatmap[frame].transpose(1, 2, 0)
                ax.imshow(heatmap_frame)
            
        ani = animation.FuncAnimation(fig, update, frames=len(self.pose_data), interval=33.3)
        
        if save_path:
            writer = animation.PillowWriter(fps=30)
            ani.save(save_path, writer=writer)

        plt.show()
       
class PoseBase:
    def __init__(self,
                 pose_path=None, 
                 pose_name=None,
                 root_folder_name=None,
                 pose_type=None,
                 feature_folder_name=None,                 
                 dataset=None,
                 file_extention=None,
                 pose_conf_threshold = 0.1,
                 **kwargs) -> None:
        
        self.pose_path=pose_path
        self.pose_name=pose_name
        self.root_folder_name=root_folder_name
        self.pose_type=pose_type
        self.feature_folder_name=feature_folder_name
        self.dataset=dataset
        self.file_extention=file_extention
        self.pose_conf_threshold = pose_conf_threshold
        self.pose_type_list = self._get_pose_type_list(pose_type)


        if pose_path is None:
            self.pose_path = self._get_file_path(
                pose_name=pose_name,
                root_folder_name=root_folder_name,
                pose_type=pose_type,
                feature_folder_name=feature_folder_name,                 
                dataset=dataset,
                file_extention=file_extention,
            )
        else:
            self.pose_path = pose_path
        
        self.pose_dict = self._load_pose(pose_path)


        self.pose_data = self._reformat_input()
        
    def _get_pose_type_list(self, pose_type=None):
        raise NotImplemented

    def _get_file_path(self,
                         pose_name,
                         root_folder_name,
                         dataset,
                         pose_type,
                         file_extention = None,
                         feature_folder_name=None):
            

        if feature_folder_name is None:
            feature_folder_name = POSE_FOLDER_NAMES_MAP[pose_type]
        if file_extention is None:
            file_extention = POSE_EXTENTION_MAP[pose_type]
        filename = f'{pose_name}.{file_extention}'
        return osp.join(root_folder_name,dataset,pose_name,feature_folder_name,filename)
    
    def _load_pose(self,pose_path):
        raise NotImplemented
    
    def _reformat_input(self):
        raise NotImplemented
 
class PoseFace(PoseBase, PoseBaseUtils):
    def __init__(self, 
                 pose_path=None,
                    pose_name=None, 
                    root_folder_name=None, 
                    pose_type=None,
                    feature_folder_name=None,
                    dataset=None,
                    file_extention=None,
                    num_joints=None, image_width=1624,
                    image_height=1224,
                    heat_map_sigma=5,
                    use_conf_as_sigma=False,
                    pose_conf_threshold=0.8,
                    create_heatmap = False,
                    **kwargs) -> None:
        self.create_heatmap = create_heatmap
        if num_joints is None:
            num_joints = POSE_NUMER_JOINTS_MAP[pose_type]
        self.num_joints = num_joints
        PoseBase.__init__(self,
                    pose_path=pose_path,
                    pose_name=pose_name,
                    root_folder_name=root_folder_name,
                    pose_type=pose_type,
                    feature_folder_name=feature_folder_name,
                    dataset=dataset,
                    file_extention=file_extention,
                    pose_conf_threshold=pose_conf_threshold)
        
        PoseBaseUtils.__init__(self,
                    pose_data = self.pose_data,
                    num_joints=num_joints,
                    image_width=image_width,
                    image_height=image_height,
                    heat_map_sigma=heat_map_sigma,
                    use_conf_as_sigma=use_conf_as_sigma,
                    create_heatmap=self.create_heatmap
                    )      
        
        if num_joints is None:
            num_joints = POSE_NUMER_JOINTS_MAP[pose_type]
        self.num_joints = num_joints
    def _get_pose_type_list(self, pose_type=None):
        if pose_type is None:
            pose_type = 'face'
        return [pose_type]
        
    def _load_pose(self, pose_path):
        try:
            if pose_path is None:
                pose_path = self.pose_path

            if isinstance(pose_path, str):
                df = pd.read_csv(pose_path)
                pose_data = df[['frame','face_id']+FACE_JOINT_NAMES_X+FACE_JOINT_NAMES_Y].to_dict()
            else:
                raise f'pose_path must be string to path, but it was {type(pose_path)}'
        except:
            print(f'There was a problem with file {pose_path}')
            return pd.DataFrame(columns=['frame', 'face_id'] + FACE_JOINT_NAMES_X + FACE_JOINT_NAMES_Y)
        return pose_data

    def _reformat_input(self):
            # Initialize the resulting dictionary
        ret = {}
        df = pd.DataFrame.from_dict(self.pose_dict)
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            try:
                if np.isnan(row['frame']):
                    continue
                if np.isnan(row['face_id']):
                    continue
                frame_id = int(row['frame']-1) # frame number should start with 0 to be compatible with others
                face_id = int(row['face_id'])
                
                
                # Extract and store the joint coordinates
                for col in df.columns:
                    if col.startswith('x_'):
                        joint_id = int(col.split('_')[1])-1
                        x = row[col]
                        y = row[col.replace('x', 'y')]
                        if not (np.isnan(x) | np.isnan(x)):
                            # Create the structure if it doesn't exist
                            if frame_id not in ret:
                                ret[frame_id] = {'face': {}}
                            if 'face' not in ret[frame_id]:
                                ret[frame_id]['face'] = {}
                            if face_id not in ret[frame_id]['face']:
                                ret[frame_id]['face'][face_id] = {}

                            ret[frame_id]['face'][face_id][joint_id] = (x, y)
            except:
                continue   
        return ret
    
class PoseBodyHand(PoseBase, PoseBaseUtils):
    def __init__(self, pose_path=None,
                    pose_name=None, 
                    root_folder_name=None, 
                    pose_type=None,
                    feature_folder_name=None,
                    dataset=None,
                    file_extention=None,
                    num_joints=None,
                    image_width=1624,
                    image_height=1224,
                    heat_map_sigma=5,
                    use_conf_as_sigma=False,
                    pose_conf_threshold=0.8,
                    create_heatmap = False,
                    **kwargs) -> None:
        self.create_heatmap = create_heatmap
        if num_joints is None:
            num_joints = POSE_NUMER_JOINTS_MAP[pose_type]
        self.num_joints = num_joints
        PoseBase.__init__(self,
                    pose_path=pose_path,
                    pose_name=pose_name,
                    root_folder_name=root_folder_name,
                    pose_type=pose_type,
                    feature_folder_name=feature_folder_name,
                    dataset=dataset,
                    file_extention=file_extention,
                    pose_conf_threshold=pose_conf_threshold)
        PoseBaseUtils.__init__(self,
                    pose_data = self.pose_data,
                    num_joints=num_joints,
                    image_width=image_width,
                    image_height=image_height,
                    heat_map_sigma=heat_map_sigma,
                    use_conf_as_sigma=use_conf_as_sigma,
                    create_heatmap=self.create_heatmap
                    )  


            
    def _get_pose_type_list(self, pose_type=None):
        if pose_type is None or pose_type=='body_hand':
            pose_type = ['body','hand']
        return pose_type

    def _load_pose(self, pose_path):
        if pose_path is None:
            pose_path = self.pose_path

        if isinstance(pose_path, str):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
        else:
            raise f'pose_path must be string to path, but it was {type(pose_path)}'
        return pose_data
    
    def _reformat_input(self):
        '''Reformat input from {frame: body: [[body list], ]}
        body list: {joint_int: [[cordx, cordy] , conf]}
        to:
        {frame: body: [body list]}
        body list: {joint_int:[cordx, cordy]}
        '''
        pose_data = self.pose_dict
        pose_conf_th = self.pose_conf_threshold
        ret = {}
        for frame_number_txt, frame_data in tqdm(pose_data.items(), total=64, disable=not VERBOSE):
            frame_number = int(frame_number_txt)
            for body_or_hands, joint_data in frame_data.items():
                if body_or_hands == 'body':
                    for person_ind, person_data in enumerate(joint_data):
                        for joint_number, (coords, confidence) in person_data[0].items():
                            x, y = coords[0], coords[1]
                            if confidence>pose_conf_th:
                                # Add values to the dictionary
                                if frame_number not in ret:
                                    ret[frame_number] = {}
                                if body_or_hands not in ret[frame_number]:
                                    ret[frame_number][body_or_hands] = {}
                                if person_ind not in ret[frame_number][body_or_hands]:
                                    ret[frame_number][body_or_hands][person_ind] = {}
                                ret[frame_number][body_or_hands][person_ind][joint_number]=(x, y)
                
                elif body_or_hands == 'hand':
                    for person_ind, person_data in enumerate(joint_data):
                        for joint_number, coords in person_data.items():
                            x, y = int(coords[0]), int(coords[1])
                            if confidence>pose_conf_th:
                                # Add values to the dictionary
                                if frame_number not in ret:
                                    ret[frame_number] = {}
                                if body_or_hands not in ret[frame_number]:
                                    ret[frame_number][body_or_hands] = {}
                                if person_ind not in ret[frame_number][body_or_hands]:
                                    ret[frame_number][body_or_hands][person_ind] = {}
                                ret[frame_number][body_or_hands][person_ind][joint_number]=(x, y)
        return ret

class PoseCollection(PoseBaseUtils):
    def __init__(self,root_folder_name,dataset,pose_name,pose_types=['face','body_hand'],image_height=None, image_width=None, **kwargs):

        self.pose_name=pose_name
        self.root_folder_name=root_folder_name
        self.pose_type=pose_types
        self.feature_folder_name=None
        self.dataset=dataset

        pose_object_list = []
        for pose_type in pose_types:
            if pose_type == 'face':
                pose_object_list.append(PoseFace(
                    pose_name=pose_name,
                    root_folder_name=root_folder_name,
                    dataset=dataset,
                    pose_type=pose_type,
                    image_height=image_height,
                    image_width=image_width,
                    **kwargs
                ))
            elif pose_type == 'body_hand':
                pose_object_list.append(PoseBodyHand(
                    pose_name=pose_name,
                    root_folder_name=root_folder_name,
                    dataset=dataset,
                    pose_type=pose_type,
                    image_height=image_height,
                    image_width=image_width,
                    **kwargs
                ))
            else:
                raise f'pose_type {pose_type} is not supported'
        self.pose_object_list = pose_object_list
        pose_data = self._get_pose_data()
        self.pose_type_list = self._get_pose_type_list()
        num_joints = self._get_num_joints()
        super().__init__(
                 pose_data = pose_data,
                 num_joints=num_joints,
                 image_width=image_width,
                 image_height=image_height,
                 heat_map_sigma=5,
                 use_conf_as_sigma=False,
                 create_heatmap=True,
        )

    def get_pose_object(self, pose_type):
        for pose_object in self.pose_object_list:
            if pose_object.pose_type == pose_type:
                return pose_object
        raise f'pose_type {pose_type} is not supported'
    def get_pose_object_list(self):
        return self.pose_object_list
    
    def _get_pose_type_list(self):
        pose_type_list = []
        for pose_object in self.pose_object_list:
            pose_type_list += pose_object.pose_type_list
        return pose_type_list
    
        
    def _get_pose_data(self):
        ret = {}
        for pose_object in self.pose_object_list:
            temp_data = pose_object.pose_data
            for frame_number, frame_data in temp_data.items():
                if frame_number not in ret:
                    ret[frame_number] = {}
                for body_part, body_part_data in frame_data.items():
                    ret[frame_number][body_part] = body_part_data
        return ret
    
    def _get_num_joints(self):
        ret = 0
        for pose_object in self.pose_object_list:
            ret += pose_object.num_joints
        return ret
    
def print_nested_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print(' ' * indent + str(key) + ':')
            print_nested_dict(value, indent + 4)
        elif isinstance(value, list):
            print(' ' * indent + str(key) + ':')
            for item in value:
                if isinstance(item, dict):
                    print_nested_dict(item, indent + 4)
                else:
                    print(' ' * (indent + 4) + str(item))
        else:
            print(' ' * indent + str(key) + ': ' + str(value))

def get_video_dimensions(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the video frame dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Release the video capture object
    cap.release()
    
    return width, height    

    # If dimensions are not found
    return None, None



def save_as_video(frames, output_file, fps=30):
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width = frames.shape[2], frames.shape[3]
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Convert frames to uint8 and write to video file
    for frame in frames:
        frame = (frame * 255).astype(np.uint8)  # Assuming the frame values are in [0, 1] range
        frame = frame.transpose(1, 2, 0)
        out.write(frame)
    out.release()

if __name__ == '__main__':
    # pose_path_body_hand = '/Users/itzikg/workspace/motion_classifier/01644-video2.json'
    # pose_path_face = '/Users/itzikg/workspace/motion_classifier/01644-video2.csv'
    # pose_obj = PoseBodyHand(pose_path=pose_path_body_hand)
    
    home_folder = '/home/ubuntu'
    root_folder_name = osp.join(home_folder,'data_local/mpi_data/2Itzik/MPIIGroupInteraction/features')
    feature_folder_name = None
    dataset = 'val'
    # pose_name = '01645-video2'
    pose_name = '01647-video2'

    # pose_obj_face = PoseFace(pose_path=None, 
    #              pose_name=pose_name,
    #              root_folder_name=root_folder_name,
    #              pose_type='face',
    #              feature_folder_name=feature_folder_name,                 
    #              dataset=dataset,
    #              file_extention='csv',              
    #              num_joints=None,
    #              image_width=1624,
    #              image_height=1224,
    #              heat_map_sigma=5,
    #              use_conf_as_sigma=False,
    #              pose_conf_threshold=0.8

    # )

    # pose_obj_hand_body = PoseBodyHand(pose_path=None, 
                #  pose_name=pose_name,
                #  root_folder_name=root_folder_name,
                #  pose_type='body_hand',
                #  feature_folder_name=feature_folder_name,                 
                #  dataset=dataset,
                #  file_extention='json',              
                #  num_joints=None,
                #  image_width=1624,
                #  image_height=1224,
                #  heat_map_sigma=5,
                #  use_conf_as_sigma=False,
                #  pose_conf_threshold=0.8
    # )

    pose_obj_collection = PoseCollection(root_folder_name=root_folder_name,
                                     dataset=dataset,
                                     pose_name=pose_name,
                                     pose_types=['face', 'body_hand'],
                                     feature_folder_name=feature_folder_name,
                                     file_extention=None,
                                     num_joints=None,
                                     image_width=None,
                                     image_height=None,
                                     heat_map_sigma=5,
                                     use_conf_as_sigma=False,
                                     pose_conf_threshold=0.8
                                     )

    # ret = pose_obj._reformat_input()
    # ret = pose_obj_face.pose_dict
    # print_nested_dict(ret)
    # pose_obj_face._draw_skeleton(scale=1)
    # pose_obj_collection._draw_skeleton(scale=1)
    # video_save_path = pose_obj_collection.video_path.replace('clips','heatmap')
    # os.makedirs(osp.dirname(video_save_path), exist_ok=True)
    # ret = pose_obj_collection._draw_skeleton(save_path=video_save_path)
    # pose_obj_collection.save_video(output_file='vid_delete.mp4')
    pose_obj_collection.save_video()
    # print(ret.shape)
    # save_as_video(frames=np.random.rand(64, 3, 10, 20), output_file='testing.mp4', fps=30)







