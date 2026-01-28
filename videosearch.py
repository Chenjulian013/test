"""
基于三个视频的完整检索系统 - 支持自定义测试图像和图片选择检测功能
视频路径：
1. D:\vscodeproject\data\search.mp4
2. D:\vscodeproject\data\search2.mp4  
3. D:\vscodeproject\data\search3.mp4
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pickle
import time
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class ThreeVideoRetrievalSystem:
    def __init__(self):
        """初始化三视频检索系统"""
        print("初始化三视频检索系统...")
        
        # 您的三个视频路径
        self.video_paths = [
            r"D:\vscodeproject\data\search.mp4",
            r"D:\vscodeproject\data\search2.mp4",
            r"D:\vscodeproject\data\search3.mp4"
        ]
        
        # 验证视频文件是否存在
        self.valid_video_paths = []
        for path in self.video_paths:
            if os.path.exists(path):
                self.valid_video_paths.append(path)
                print(f"✓ 找到视频: {os.path.basename(path)}")
            else:
                print(f"✗ 视频不存在: {os.path.basename(path)}")
        
        if len(self.valid_video_paths) == 0:
            print("没有找到有效的视频文件，程序退出")
            exit(1)
        
        # 设置设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载预训练模型
        self.model = models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # 存储数据
        self.video_features = []
        self.video_names = []
        self.video_info = []
        
        # 关键帧数据库
        self.keyframe_features = []
        self.keyframe_video_ids = []
        self.keyframe_timestamps = []
        
    def extract_features(self, image):
        """从图像提取特征"""
        try:
            # 转换为RGB
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 预处理
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # 提取特征
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            # 归一化
            if np.linalg.norm(features) > 0:
                features = features / np.linalg.norm(features)
            
            return features
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def extract_video_features(self, video_path, video_id):
        """提取视频关键帧特征"""
        print(f"\n处理视频 {video_id+1}: {os.path.basename(video_path)}")
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  无法打开视频")
            return [], []
        
        # 获取视频信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # 存储视频信息
        video_name = os.path.basename(video_path)
        self.video_info.append({
            'id': video_id,
            'name': video_name,
            'path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'duration': duration,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        })
        
        # 提取关键帧（每1秒或每30帧取一帧）
        frame_interval = int(fps) if fps > 0 else 30
        if frame_interval == 0:
            frame_interval = 30
        
        # 限制最多提取的关键帧数
        max_keyframes = min(50, total_frames // frame_interval)
        if max_keyframes == 0:
            max_keyframes = 10
        
        frame_features = []
        timestamps = []
        frame_count = 0
        keyframe_count = 0
        
        print(f"  视频信息: {total_frames}帧, {duration:.1f}秒, {fps:.1f}FPS")
        print(f"  关键帧间隔: {frame_interval}帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 每frame_interval帧提取一个关键帧
            if frame_count % frame_interval == 0 and keyframe_count < max_keyframes:
                # 计算时间戳
                timestamp = frame_count / fps if fps > 0 else 0
                
                # 提取特征
                features = self.extract_features(frame)
                if features is not None:
                    frame_features.append(features)
                    timestamps.append(timestamp)
                    
                    # 添加到全局数据库
                    self.keyframe_features.append(features)
                    self.keyframe_video_ids.append(video_id)
                    self.keyframe_timestamps.append(timestamp)
                    
                    keyframe_count += 1
            
            frame_count += 1
            
            # 显示进度
            if frame_count % 500 == 0:
                print(f"    已处理 {frame_count}/{total_frames} 帧，提取 {keyframe_count} 个关键帧")
        
        cap.release()
        
        # 计算视频的整体特征
        if len(frame_features) > 0:
            video_avg_features = np.mean(frame_features, axis=0)
            video_avg_features = video_avg_features / np.linalg.norm(video_avg_features)
        else:
            video_avg_features = None
        
        print(f"  完成! 提取了 {keyframe_count} 个关键帧")
        
        return video_avg_features, keyframe_count
    
    def build_database(self):
        """构建视频数据库"""
        print("\n" + "="*60)
        print("开始构建视频数据库")
        print("="*60)
        
        total_keyframes = 0
        
        for i, video_path in enumerate(self.valid_video_paths):
            video_features, keyframe_count = self.extract_video_features(video_path, i)
            
            if video_features is not None:
                self.video_features.append(video_features)
                self.video_names.append(os.path.basename(video_path))
                total_keyframes += keyframe_count
        
        # 转换为numpy数组
        if len(self.keyframe_features) > 0:
            self.keyframe_features = np.array(self.keyframe_features)
            self.keyframe_video_ids = np.array(self.keyframe_video_ids)
            self.keyframe_timestamps = np.array(self.keyframe_timestamps)
        
        print(f"\n数据库构建完成!")
        print(f"  视频数量: {len(self.video_features)}")
        print(f"  总关键帧数: {total_keyframes}")
        
        # 保存数据库
        self.save_database()
        
        return True
    
    def save_database(self):
        """保存数据库到文件"""
        database = {
            'video_features': self.video_features,
            'video_names': self.video_names,
            'video_info': self.video_info,
            'keyframe_features': self.keyframe_features,
            'keyframe_video_ids': self.keyframe_video_ids,
            'keyframe_timestamps': self.keyframe_timestamps
        }
        
        with open('video_database.pkl', 'wb') as f:
            pickle.dump(database, f)
        
        print("数据库已保存到 video_database.pkl")
    
    def load_database(self):
        """从文件加载数据库"""
        if os.path.exists('video_database.pkl'):
            print("加载已保存的数据库...")
            with open('video_database.pkl', 'rb') as f:
                database = pickle.load(f)
            
            self.video_features = database['video_features']
            self.video_names = database['video_names']
            self.video_info = database['video_info']
            self.keyframe_features = database['keyframe_features']
            self.keyframe_video_ids = database['keyframe_video_ids']
            self.keyframe_timestamps = database['keyframe_timestamps']
            
            print(f"  已加载 {len(self.video_names)} 个视频的数据库")
            print(f"  总关键帧数: {len(self.keyframe_features)}")
            return True
        else:
            print("没有找到已保存的数据库")
            return False
    
    def search_videos(self, query_image_path, top_k=10):
        """搜索三个视频中相似的场景"""
        print(f"\n搜索查询图像: {os.path.basename(query_image_path)}")
        
        # 检查查询图像是否存在
        if not os.path.exists(query_image_path):
            print(f"查询图像不存在: {query_image_path}")
            return [], 0
        
        # 加载查询图像
        try:
            query_image = Image.open(query_image_path)
        except Exception as e:
            print(f"无法加载图像: {e}")
            return [], 0
        
        # 计时开始
        start_time = time.time()
        
        # 提取查询图像特征
        query_features = self.extract_features(query_image)
        if query_features is None:
            print("无法提取查询图像特征")
            return [], 0
        
        # 计算与所有关键帧的相似度
        if len(self.keyframe_features) == 0:
            print("数据库中没有关键帧")
            return [], 0
        
        # 批量计算余弦相似度
        similarities = np.dot(self.keyframe_features, query_features)
        
        # 获取前top_k个结果
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 计算搜索时间
        search_time = time.time() - start_time
        
        # 组织结果
        results = []
        for i, idx in enumerate(top_indices):
            video_id = self.keyframe_video_ids[idx]
            video_name = self.video_names[video_id]
            timestamp = self.keyframe_timestamps[idx]
            
            # 转换为时分秒格式
            minutes = int(timestamp // 60)
            seconds = int(timestamp % 60)
            
            results.append({
                'rank': i + 1,
                'video_id': int(video_id),
                'video_name': video_name,
                'similarity': float(similarities[idx]),
                'timestamp': timestamp,
                'time_str': f"{minutes:02d}:{seconds:02d}",
                'keyframe_index': int(idx)
            })
        
        return results, search_time
    
    def evaluate_precision_at_k(self, test_queries, k_values=[1, 5, 10]):
        """评估准确率@K (Precision@K)"""
        print("\n" + "="*60)
        print("评估准确率@K (Precision@K)")
        print("="*60)
        
        precision_results = {k: [] for k in k_values}
        
        for i, (query_path, ground_truth_video_id) in enumerate(test_queries):
            if not os.path.exists(query_path):
                continue
            
            print(f"\n测试查询 {i+1}: {os.path.basename(query_path)}")
            print(f"  真实视频: {self.video_names[ground_truth_video_id]}")
            
            # 执行搜索
            results, _ = self.search_videos(query_path, top_k=max(k_values))
            
            if not results:
                print("  没有搜索结果")
                continue
            
            # 计算每个k值的准确率
            for k in k_values:
                top_k_results = results[:k]
                
                # 计算准确率：前k个结果中属于正确视频的比例
                correct_count = sum(1 for r in top_k_results if r['video_id'] == ground_truth_video_id)
                precision = correct_count / k
                precision_results[k].append(precision)
                
                print(f"  Precision@{k}: {precision:.3f} ({correct_count}/{k})")
        
        # 计算平均准确率
        avg_precision = {}
        for k in k_values:
            if precision_results[k]:
                avg_precision[k] = np.mean(precision_results[k])
            else:
                avg_precision[k] = 0
        
        print("\n平均准确率@K:")
        for k in k_values:
            print(f"  P@{k}: {avg_precision[k]:.4f}")
        
        return avg_precision
    
    def evaluate_map_at_k(self, test_queries, k=10):
        """评估平均精度均值@K (mAP@K)"""
        print("\n" + "="*60)
        print(f"评估平均精度均值@K (mAP@{k})")
        print("="*60)
        
        ap_scores = []
        
        for i, (query_path, ground_truth_video_id) in enumerate(test_queries):
            if not os.path.exists(query_path):
                continue
            
            print(f"\n测试查询 {i+1}: {os.path.basename(query_path)}")
            print(f"  真实视频: {self.video_names[ground_truth_video_id]}")
            
            # 执行搜索
            results, _ = self.search_videos(query_path, top_k=k)
            
            if not results:
                print("  没有搜索结果")
                ap_scores.append(0)
                continue
            
            # 计算平均精度 (AP)
            relevant_count = 0
            precision_sum = 0
            
            for rank, result in enumerate(results, 1):
                if result['video_id'] == ground_truth_video_id:
                    relevant_count += 1
                    precision_at_rank = relevant_count / rank
                    precision_sum += precision_at_rank
            
            # 计算AP
            if relevant_count > 0:
                ap = precision_sum / relevant_count
            else:
                ap = 0
            
            ap_scores.append(ap)
            print(f"  AP: {ap:.4f} (相关结果: {relevant_count}/{k})")
        
        # 计算mAP
        if ap_scores:
            mAP = np.mean(ap_scores)
            print(f"\n平均精度均值@K (mAP@{k}): {mAP:.4f}")
        else:
            mAP = 0
            print(f"\n无法计算mAP，没有有效的测试查询")
        
        return mAP
    
    def evaluate_search_efficiency(self, test_queries, num_runs=5):
        """评估搜索效率"""
        print("\n" + "="*60)
        print("评估搜索效率")
        print("="*60)
        
        search_times = []
        
        for run in range(num_runs):
            print(f"\n运行 {run+1}/{num_runs}:")
            
            run_times = []
            for i, (query_path, _) in enumerate(test_queries[:3]):  # 只测试前3个查询
                if not os.path.exists(query_path):
                    continue
                
                # 执行搜索并计时
                _, search_time = self.search_videos(query_path, top_k=10)
                run_times.append(search_time)
                
                print(f"  查询 {i+1}: {search_time:.4f}秒")
            
            if run_times:
                avg_run_time = np.mean(run_times)
                search_times.append(avg_run_time)
                print(f"  平均搜索时间: {avg_run_time:.4f}秒")
        
        if search_times:
            avg_search_time = np.mean(search_times)
            std_search_time = np.std(search_times)
            
            print(f"\n效率评估结果:")
            print(f"  平均搜索时间: {avg_search_time:.4f}秒")
            print(f"  搜索时间标准差: {std_search_time:.4f}秒")
            print(f"  总关键帧数: {len(self.keyframe_features)}")
            print(f"  搜索速度: {len(self.keyframe_features)/avg_search_time:.0f} 关键帧/秒")
            
            return avg_search_time, std_search_time
        else:
            print("无法评估搜索效率")
            return 0, 0
    
    def create_test_queries_auto(self, num_queries_per_video=3):
        """自动创建测试查询（从每个视频中随机截取帧）"""
        print("\n自动创建测试查询...")
        
        test_queries = []
        
        for video_id, video_info in enumerate(self.video_info):
            video_path = video_info['path']
            total_frames = video_info['total_frames']
            
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue
            
            # 随机选择帧
            frame_indices = random.sample(range(0, total_frames, 100), 
                                         min(num_queries_per_video, total_frames//100))
            
            for i, frame_idx in enumerate(frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    # 保存测试图像
                    query_name = f"test_query_v{video_id+1}_{i+1}.jpg"
                    query_path = os.path.join("test_queries", query_name)
                    
                    # 确保目录存在
                    os.makedirs("test_queries", exist_ok=True)
                    
                    cv2.imwrite(query_path, frame)
                    test_queries.append((query_path, video_id))
                    
                    print(f"  创建: {query_name} (来自 {self.video_names[video_id]})")
            
            cap.release()
        
        print(f"共创建了 {len(test_queries)} 个测试查询")
        return test_queries
    
    def load_custom_test_queries(self, test_config):
        """加载自定义测试查询"""
        print("\n加载自定义测试查询...")
        
        test_queries = []
        
        for query_config in test_config:
            query_path = query_config['path']
            video_name = query_config['video_name']
            
            # 查找对应的视频ID
            video_id = -1
            for i, name in enumerate(self.video_names):
                if name == video_name:
                    video_id = i
                    break
            
            if video_id == -1:
                print(f"  警告: 找不到视频 '{video_name}'，跳过查询 {os.path.basename(query_path)}")
                continue
            
            if os.path.exists(query_path):
                test_queries.append((query_path, video_id))
                print(f"  加载: {os.path.basename(query_path)} (属于 {video_name})")
            else:
                print(f"  警告: 查询图像不存在 {query_path}")
        
        print(f"共加载了 {len(test_queries)} 个测试查询")
        return test_queries
    
    def display_search_results(self, query_image_path, results, search_time):
        """显示搜索结果"""
        print("\n" + "="*60)
        print("搜索结果")
        print("="*60)
        
        # 显示查询图像信息
        try:
            query_img = Image.open(query_image_path)
            print(f"查询图像: {os.path.basename(query_image_path)}")
            print(f"图像尺寸: {query_img.size}")
        except:
            print(f"查询图像: {os.path.basename(query_image_path)}")
        
        print(f"搜索时间: {search_time:.4f}秒")
        print(f"返回结果数: {len(results)}")
        print()
        
        # 按视频分组显示结果
        video_results = defaultdict(list)
        for result in results:
            video_results[result['video_id']].append(result)
        
        for video_id in sorted(video_results.keys()):
            video_name = self.video_names[video_id]
            video_results_list = video_results[video_id]
            
            print(f"视频: {video_name}")
            print("-" * 40)
            
            for result in video_results_list:
                print(f"  排名 {result['rank']}:")
                print(f"    相似度: {result['similarity']:.4f}")
                print(f"    时间位置: {result['time_str']} (第 {result['timestamp']:.1f} 秒)")
                print()
    
    def visualize_results(self, query_image_path, results, top_n=5):
        """可视化搜索结果"""
        if not results:
            print("没有结果可以可视化")
            return
        
        # 创建图形
        fig, axes = plt.subplots(2, top_n, figsize=(15, 6))
        
        # 显示查询图像
        ax = axes[0, 0] if top_n > 1 else axes[0]
        try:
            query_img = Image.open(query_image_path)
            ax.imshow(query_img)
            ax.set_title('查询图像', fontsize=10)
        except:
            ax.text(0.5, 0.5, '无法加载查询图像', 
                   horizontalalignment='center',
                   verticalalignment='center',
                   transform=ax.transAxes)
            ax.set_title('查询图像', fontsize=10)
        ax.axis('off')
        
        # 显示前top_n个结果
        for i in range(1, top_n):
            if i < len(results):
                result = results[i-1]
                video_path = self.video_info[result['video_id']]['path']
                
                # 打开视频并提取对应帧
                cap = cv2.VideoCapture(video_path)
                if cap.isOpened():
                    # 跳转到对应时间点
                    target_frame = int(result['timestamp'] * self.video_info[result['video_id']]['fps'])
                    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        ax = axes[0, i] if top_n > 1 else axes[1]
                        ax.imshow(frame_rgb)
                        title = f"排名 {result['rank']}\n{result['video_name']}\n{result['time_str']}\n相似度: {result['similarity']:.3f}"
                        ax.set_title(title, fontsize=8)
                        ax.axis('off')
        
        plt.suptitle('视频检索结果可视化', fontsize=14)
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("="*70)
    print("基于三个视频的完整检索系统")
    print("="*70)
    
    # 创建检索系统
    retrieval_system = ThreeVideoRetrievalSystem()
    
    # 尝试加载已有数据库，否则构建新数据库
    if not retrieval_system.load_database():
        print("\n需要构建新的视频数据库...")
        retrieval_system.build_database()
    
    # ===== 配置自定义测试查询 =====
    # 配置您的测试图像和它们对应的视频
    # 格式: {'path': '图像路径', 'video_name': '所属视频文件名'}
    
    test_config = [
        # 示例配置 - 请根据您的实际测试图像修改
        # 每个测试图像应该属于哪个视频？
        {'path': r'D:\vscodeproject\data\test_images\test1.png', 'video_name': 'search.mp4'},
        {'path': r'D:\vscodeproject\data\test_images\test2.png', 'video_name': 'search.mp4'},
        {'path': r'D:\vscodeproject\data\test_images\test3.png', 'video_name': 'search2.mp4'},
        {'path': r'D:\vscodeproject\data\test_images\test4.png', 'video_name': 'search2.mp4'},
        {'path': r'D:\vscodeproject\data\test_images\test5.png', 'video_name': 'search3.mp4'},
        {'path': r'D:\vscodeproject\data\test_images\test6.png', 'video_name': 'search3.mp4'},
    ]
    
    # 选择测试查询模式：
    # 1. 使用自定义测试查询（需要您配置上面的test_config）
    # 2. 自动生成测试查询（从视频中随机截取）
    
    test_mode = 1  # 1=自定义测试，2=自动生成
    
    if test_mode == 1:
        # 使用自定义测试查询
        test_queries = retrieval_system.load_custom_test_queries(test_config)
    else:
        # 自动生成测试查询
        test_queries = retrieval_system.create_test_queries_auto(num_queries_per_video=2)
    
    if not test_queries:
        print("\n没有测试查询，使用默认测试...")
        # 创建一个简单的测试图像
        test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128
        cv2.putText(test_img, "测试图像", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        test_path = "default_test.jpg"
        cv2.imwrite(test_path, test_img)
        test_queries = [(test_path, 0)]  # 假设属于第一个视频
    
    # 性能评估
    print("\n" + "="*70)
    print("性能评估")
    print("="*70)
    
    # 评估1: 准确率@K
    print("\n>>> 评估指标1: 准确率@K")
    precision_results = retrieval_system.evaluate_precision_at_k(test_queries, k_values=[1, 3, 5, 10])
    
    # 评估2: 平均精度均值@K
    print("\n>>> 评估指标2: 平均精度均值@K")
    map_score = retrieval_system.evaluate_map_at_k(test_queries, k=10)
    
    # 评估3: 搜索效率
    print("\n>>> 评估指标3: 搜索效率")
    avg_search_time, std_search_time = retrieval_system.evaluate_search_efficiency(test_queries, num_runs=3)
    
    # 总结报告
    print("\n" + "="*70)
    print("性能评估总结")
    print("="*70)
    print(f"视频数量: {len(retrieval_system.video_names)}")
    print(f"总关键帧数: {len(retrieval_system.keyframe_features)}")
    print(f"测试查询数: {len(test_queries)}")
    print()
    print("准确率@K:")
    for k, p in precision_results.items():
        print(f"  P@{k}: {p:.4f}")
    print(f"平均精度均值@10: {map_score:.4f}")
    print(f"平均搜索时间: {avg_search_time:.4f}秒 (±{std_search_time:.4f})")
    
    # 生成性能报告
    generate_performance_report(retrieval_system, precision_results, map_score, avg_search_time, std_search_time)
    
    # 交互式搜索
    print("\n" + "="*70)
    print("交互式搜索模式")
    print("="*70)
    print("输入图像路径进行搜索，或输入以下命令:")
    print("  'select' - 选择一张测试图片进行检测")
    print("  'test' - 使用第一个测试查询")
    print("  'random' - 从视频中随机截取测试图像")
    print("  'custom' - 输入自定义查询图像")
    print("  'evaluate' - 重新运行性能评估")
    print("  'list' - 列出所有视频")
    print("  'report' - 查看性能报告")
    print("  'exit' - 退出程序")
    
    while True:
        user_input = input("\n请输入命令或图像路径: ").strip()
        
        if user_input.lower() == 'exit':
            print("退出程序")
            break
        
        elif user_input.lower() == 'select':
            # 选择一张测试图片进行检测
            if test_queries:
                print("\n可用的测试图片:")
                for i, (query_path, video_id) in enumerate(test_queries):
                    print(f"  {i+1}. {os.path.basename(query_path)} (属于 {retrieval_system.video_names[video_id]})")
                
                try:
                    choice = int(input("\n请选择图片编号: ")) - 1
                    if 0 <= choice < len(test_queries):
                        query_path, true_video_id = test_queries[choice]
                        true_video_name = retrieval_system.video_names[true_video_id]
                        
                        print(f"\n您选择了: {os.path.basename(query_path)}")
                        print(f"真实所属视频: {true_video_name}")
                        
                        # 执行搜索
                        results, search_time = retrieval_system.search_videos(query_path, top_k=10)
                        
                        if results:
                            # 显示搜索结果
                            retrieval_system.display_search_results(query_path, results, search_time)
                            
                            # 检测视频归属
                            print("\n" + "="*60)
                            print("视频归属检测结果")
                            print("="*60)
                            
                            # 统计各个视频的出现次数
                            video_counts = {}
                            for result in results:
                                video_id = result['video_id']
                                video_counts[video_id] = video_counts.get(video_id, 0) + 1
                            
                            # 按出现次数排序
                            sorted_videos = sorted(video_counts.items(), key=lambda x: x[1], reverse=True)
                            
                            print("检测到的视频概率分布:")
                            total_results = len(results)
                            for video_id, count in sorted_videos:
                                video_name = retrieval_system.video_names[video_id]
                                percentage = count / total_results * 100
                                print(f"  {video_name}: {count}次 ({percentage:.1f}%)")
                            
                            # 预测最可能的视频
                            predicted_video_id = sorted_videos[0][0] if sorted_videos else -1
                            predicted_video_name = retrieval_system.video_names[predicted_video_id]
                            predicted_count = sorted_videos[0][1] if sorted_videos else 0
                            predicted_percentage = predicted_count / total_results * 100
                            
                            print(f"\n预测结果: 最可能属于视频 '{predicted_video_name}'")
                            print(f"  出现次数: {predicted_count}/{total_results}")
                            print(f"  置信度: {predicted_percentage:.1f}%")
                            
                            # 检查预测是否正确
                            if predicted_video_id == true_video_id:
                                print(f"  ✓ 预测正确! (真实视频: {true_video_name})")
                            else:
                                print(f"  ✗ 预测错误! (真实视频: {true_video_name})")
                            
                            # 详细分析前3个结果
                            print("\n前3个最相似场景:")
                            for i, result in enumerate(results[:3]):
                                print(f"  {i+1}. 时间: {result['time_str']}, 相似度: {result['similarity']:.4f}, 视频: {result['video_name']}")
                            
                            # 询问是否可视化
                            viz_choice = input("\n是否可视化结果？(y/n): ").strip().lower()
                            if viz_choice == 'y':
                                retrieval_system.visualize_results(query_path, results, top_n=5)
                        else:
                            print("没有找到相关视频")
                    else:
                        print("选择超出范围")
                except ValueError:
                    print("请输入有效的数字")
            else:
                print("没有可用的测试图片")
        
        elif user_input.lower() == 'test':
            # 使用第一个测试查询
            if test_queries:
                query_path, _ = test_queries[0]
                print(f"使用测试查询: {os.path.basename(query_path)}")
                
                # 执行搜索
                results, search_time = retrieval_system.search_videos(query_path, top_k=10)
                
                if results:
                    retrieval_system.display_search_results(query_path, results, search_time)
                    retrieval_system.visualize_results(query_path, results, top_n=5)
                else:
                    print("没有找到相关视频")
            else:
                print("没有可用的测试查询")
        
        elif user_input.lower() == 'random':
            # 从随机视频中截取测试图像
            video_id = random.randint(0, len(retrieval_system.video_info)-1)
            video_info = retrieval_system.video_info[video_id]
            
            cap = cv2.VideoCapture(video_info['path'])
            if cap.isOpened():
                # 随机选择一帧
                random_frame = random.randint(0, video_info['total_frames']-1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    random_img_path = f"random_query_v{video_id+1}.jpg"
                    cv2.imwrite(random_img_path, frame)
                    print(f"已创建随机测试图像: {random_img_path}")
                    print(f"来自视频: {video_info['name']}")
                    
                    # 执行搜索
                    results, search_time = retrieval_system.search_videos(random_img_path, top_k=10)
                    
                    if results:
                        retrieval_system.display_search_results(random_img_path, results, search_time)
                        retrieval_system.visualize_results(random_img_path, results, top_n=5)
                    else:
                        print("没有找到相关视频")
        
        elif user_input.lower() == 'custom':
            # 输入自定义查询图像
            custom_path = input("请输入查询图像路径: ").strip()
            if os.path.exists(custom_path) and custom_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 执行搜索
                results, search_time = retrieval_system.search_videos(custom_path, top_k=10)
                
                if results:
                    retrieval_system.display_search_results(custom_path, results, search_time)
                    retrieval_system.visualize_results(custom_path, results, top_n=5)
                    
                    # 让用户选择这个查询属于哪个视频（用于评估）
                    print("\n这个查询图像属于哪个视频？")
                    for i, video_name in enumerate(retrieval_system.video_names):
                        print(f"  {i+1}. {video_name}")
                    
                    try:
                        video_choice = int(input("请输入视频编号: ")) - 1
                        if 0 <= video_choice < len(retrieval_system.video_names):
                            # 添加到测试查询
                            test_queries.append((custom_path, video_choice))
                            print(f"已添加到测试查询，属于 {retrieval_system.video_names[video_choice]}")
                    except:
                        print("无效输入")
                else:
                    print("没有找到相关视频")
            else:
                print("文件不存在或不是图像文件")
        
        elif user_input.lower() == 'evaluate':
            # 重新运行性能评估
            print("重新运行性能评估...")
            precision_results = retrieval_system.evaluate_precision_at_k(test_queries, k_values=[1, 3, 5, 10])
            map_score = retrieval_system.evaluate_map_at_k(test_queries, k=10)
            avg_search_time, std_search_time = retrieval_system.evaluate_search_efficiency(test_queries, num_runs=2)
        
        elif user_input.lower() == 'list':
            # 列出所有视频
            print("\n视频列表:")
            for i, video_name in enumerate(retrieval_system.video_names):
                video_info = retrieval_system.video_info[i]
                print(f"  {i+1}. {video_name}")
                print(f"     时长: {video_info['duration']:.1f}秒, 分辨率: {video_info['width']}x{video_info['height']}")
                print(f"     帧率: {video_info['fps']:.1f}FPS, 总帧数: {video_info['total_frames']}")
                print(f"     关键帧数: {np.sum(retrieval_system.keyframe_video_ids == i)}")
        
        elif user_input.lower() == 'report':
            # 查看性能报告
            print("\n" + "="*70)
            print("性能报告")
            print("="*70)
            print(f"测试时间: {time.strftime('%Y-%m-d %H:%M:%S')}")
            print(f"视频数量: {len(retrieval_system.video_names)}")
            print(f"总关键帧数: {len(retrieval_system.keyframe_features)}")
            print(f"测试查询数: {len(test_queries)}")
            print("\n评估结果:")
            print("  - 准确率@1: {:.2%}".format(precision_results[1]))
            print("  - 准确率@3: {:.2%}".format(precision_results[3]))
            print("  - 准确率@5: {:.2%}".format(precision_results[5]))
            print("  - 准确率@10: {:.2%}".format(precision_results[10]))
            print("  - mAP@10: {:.2%}".format(map_score))
            print("  - 平均搜索时间: {:.3f}秒".format(avg_search_time))
            print("  - 搜索标准差: {:.3f}秒".format(std_search_time))
        
        elif os.path.exists(user_input):
            # 用户输入了图像路径
            if user_input.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # 执行搜索
                results, search_time = retrieval_system.search_videos(user_input, top_k=10)
                
                if results:
                    retrieval_system.display_search_results(user_input, results, search_time)
                    
                    # 分析结果
                    print("\n结果分析:")
                    # 统计每个视频的结果数量
                    video_counts = {}
                    for result in results:
                        video_id = result['video_id']
                        video_counts[video_id] = video_counts.get(video_id, 0) + 1
                    
                    print("  各视频命中数:")
                    for video_id, count in video_counts.items():
                        video_name = retrieval_system.video_names[video_id]
                        percentage = count / len(results) * 100
                        print(f"    {video_name}: {count} 个结果 ({percentage:.1f}%)")
                    
                    # 计算平均相似度
                    avg_similarity = np.mean([r['similarity'] for r in results])
                    print(f"  平均相似度: {avg_similarity:.4f}")
                    
                    # 询问是否可视化
                    viz_choice = input("\n是否可视化结果？(y/n): ").strip().lower()
                    if viz_choice == 'y':
                        retrieval_system.visualize_results(user_input, results, top_n=5)
                else:
                    print("没有找到相关视频")
            else:
                print("请提供图像文件（jpg, png, bmp）")
        else:
            print("文件不存在或命令无效")
            print("可用命令: 'select', 'test', 'random', 'custom', 'evaluate', 'list', 'report', 'exit'")

def generate_performance_report(retrieval_system, precision_results, map_score, avg_search_time, std_search_time):
    """生成性能报告文件"""
    report_content = f"""
视频检索系统性能评估报告
============================
测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}

系统配置
--------
- 使用模型: ResNet50
- 特征维度: {retrieval_system.keyframe_features.shape[1] if len(retrieval_system.keyframe_features) > 0 else 'N/A'}
- 设备: {retrieval_system.device}

视频数据库
--------
"""
    
    for i, video_name in enumerate(retrieval_system.video_names):
        video_info = retrieval_system.video_info[i]
        keyframe_count = np.sum(retrieval_system.keyframe_video_ids == i)
        report_content += f"- {video_name}: {video_info['duration']:.1f}秒, {keyframe_count}个关键帧\n"
    
    report_content += f"""
评估结果
--------
准确率@K:
- 准确率@1: {precision_results.get(1, 0):.4f} ({precision_results.get(1, 0)*100:.1f}%)
- 准确率@3: {precision_results.get(3, 0):.4f} ({precision_results.get(3, 0)*100:.1f}%)
- 准确率@5: {precision_results.get(5, 0):.4f} ({precision_results.get(5, 0)*100:.1f}%)
- 准确率@10: {precision_results.get(10, 0):.4f} ({precision_results.get(10, 0)*100:.1f}%)

平均精度均值@10 (mAP): {map_score:.4f} ({map_score*100:.1f}%)

搜索效率
- 平均搜索时间: {avg_search_time:.4f}秒
- 搜索时间标准差: {std_search_time:.4f}秒
- 总关键帧数: {len(retrieval_system.keyframe_features)}
- 搜索速度: {len(retrieval_system.keyframe_features)/avg_search_time if avg_search_time > 0 else 0:.0f} 关键帧/秒

性能分析
--------
"""
    
    # 性能分析
    if precision_results.get(1, 0) > 0.8:
        report_content += "- 准确率@1较高，系统能够准确找到最相关的视频\n"
    else:
        report_content += "- 准确率@1有待提高，可能需要优化特征提取\n"
    
    if map_score > 0.7:
        report_content += "- mAP@10较高，系统整体检索质量良好\n"
    else:
        report_content += "- mAP@10较低，系统排序质量有待提高\n"
    
    if avg_search_time < 0.1:
        report_content += "- 搜索效率很高，满足实时性要求\n"
    else:
        report_content += "- 搜索效率一般，可能需要优化搜索算法\n"
    
    report_content += """
改进建议
--------
1. 增加关键帧数量以提高检索精度
2. 使用更强大的预训练模型（如ResNet101）
3. 考虑使用注意力机制改进特征提取
4. 实现更高效的索引结构（如Faiss）
"""

    # 保存报告
    with open('performance_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"性能报告已保存到 performance_report.txt")

if __name__ == "__main__":
    # 设置matplotlib字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n程序运行时发生错误: {e}")
        import traceback
        traceback.print_exc()