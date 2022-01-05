from icevision.all import *

class ReefParser(Parser):
    def __init__(self, 
                 template_record, 
                 data_dir, 
                 annotations_csv):
        super().__init__(template_record=template_record)

        self.data_dir = data_dir
        
        # Ensure columns are in correct order so we can use name=None in itertuples -> faster
        self.df = annotations_csv
        
        self.class_map = ClassMap(['starfish'])

    def __iter__(self) -> Any:
        for o in self.df.itertuples():
            yield o

    def __len__(self) -> int:
        return len(self.df)
    
    def record_id(self, o) -> Hashable:
        return str(o.video_id) + '_' + str(o.video_frame) # need a distinct id for each image

    def parse_fields(self, o, record, is_new):
        
        if is_new:
            record.set_filepath(os.path.join(self.data_dir, f'video_{o.video_id}', f'{o.video_frame}.jpg'))
            record.set_img_size(ImgSize(width=o.img_width, height=o.img_height))
            record.detection.set_class_map(self.class_map)
        
        if not ((o.xmin == -1) and (o.ymin == -1) and (o.bb_width == -1) and (o.bb_height == -1)):
            xmin, ymin, bb_width, bb_height = o.xmin, o.ymin, o.bb_width, o.bb_height
            xmax = xmin + bb_width
            ymax = ymin + bb_height

#             record.detection.add_bboxes([BBox.from_xywh(xmin, ymin, bb_width, bb_height)])
            record.detection.add_bboxes([BBox.from_xyxy(xmin, ymin, xmax, ymax)])
            record.detection.add_labels(['starfish'])