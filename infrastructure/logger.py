import os
from tensorboardX import SummaryWriter
import numpy as np

class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=None):
        self._log_dir = log_dir
        print('###########################')
        print('logging outputs to ', log_dir)
        print('###########################')
        self._n_logged_samples = n_logged_samples
        self._summary_writer = SummaryWriter(log_dir, flush_secs=1, max_queue=1) if summary_writer is None else summary_writer
    
    def flush(self):
        self._summary_writer.flush()

    def dump_scalars(self, log_path=None):
        log_path = os.path.join(self._log_dir, 'scalar_data.json') if log_path is None else log_path
        self._summary_writer.export_scalars_to_json(log_path)

    def log_scalar(self, scalar, name, step):
        self._summary_writer.add_scalar('{}'.format(name), scalar, step)
    
    def log_scalars(self, scalar_dict, group_name, step, phase):
        self._summary_writer.add_scalars('{}/{}'.format(group_name, phase), scalar_dict, step)
    
    def log_image(self, image, name, step):
        assert(len(image.shape) == 3)
        self._summary_writer.add_image('{}'.format(name), image, step)
    
    def log_video(self, video_frames, name, step, fps=20):
        assert(len(video_frames.shape) == 5)
        self._summary_writer.add_video('{}'.format(name), video_frames, step, fps=fps)

    def log_paths_as_videos(self, paths, step, max_videos_to_save=2, fps=10, video_title='video'):
        videos = [np.transpose(p['image_obs'], [0,3,1,2]) for p in paths]
        max_videos_to_save = np.min([max_videos_to_save, len(videos)])
        videos = videos[:max_videos_to_save]
        max_length = np.max([video.shape[0] for video in videos])


        for i in range(max_videos_to_save):
            print(videos[i].shape)
            if videos[i].shape[0] < max_length:
                pad = np.tile(videos[i][-1], [max_length - videos[i].shape[0], 1, 1, 1])
                videos[i] = np.concatenate([videos[i], pad], axis=0)
        
        videos = np.stack(videos, axis=0)
        self.log_video(videos, video_title, step, fps=fps)
    
    def log_figure(self, figure, name, step, phase):
        self._summary_writer.add_figure('{}/{}'.format(name, phase), figure, step)
    
    def log_figures(self, figure, name, step, phase):
        assert figure.shape[0] > 0, 'No figures to log'
        self._summary_writer.add_figure('{}/{}'.format(name, phase), figure, step)