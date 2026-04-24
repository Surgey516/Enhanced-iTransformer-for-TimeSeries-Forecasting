import optuna
import matplotlib.pyplot as plt
import os

# Optuna 数据库路径（根据你的 tune.py 设置）
STUDY_NAME_PREFIX = 'iTransformer'
DATASETS = ['ETTm1', 'ETTm2', 'ETTh1', 'ETTh2']

def visualize_optuna_study(dataset):
    """
    为指定数据集生成 Optuna 可视化图
    """
    study_name = f'{STUDY_NAME_PREFIX}_{dataset}'
    storage = f'sqlite:///{study_name}.db'
    
    try:
        # 加载 Optuna study
        study = optuna.load_study(study_name=study_name, storage=storage)
        print(f'\n{dataset} - 成功加载 study，共 {len(study.trials)} 次试验')
        
        # 创建输出目录
        output_dir = f'Results/Optuna_{dataset}'
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 超参数重要性图
        try:
            fig = optuna.visualization.plot_param_importances(study)
            fig.write_image(f'{output_dir}/hyperparameter_importance.png', width=800, height=600)
            print(f'  ✓ 超参数重要性图已保存')
        except Exception as e:
            print(f'  ✗ 超参数重要性图生成失败: {e}')
        
        # 2. 平行坐标图
        try:
            fig = optuna.visualization.plot_parallel_coordinate(study)
            fig.write_image(f'{output_dir}/parallel_coordinate.png', width=1200, height=600)
            print(f'  ✓ 平行坐标图已保存')
        except Exception as e:
            print(f'  ✗ 平行坐标图生成失败: {e}')
        
        # 3. 优化历史图
        try:
            fig = optuna.visualization.plot_optimization_history(study)
            fig.write_image(f'{output_dir}/optimization_history.png', width=800, height=600)
            print(f'  ✓ 优化历史图已保存')
        except Exception as e:
            print(f'  ✗ 优化历史图生成失败: {e}')
        
        # 4. 切片图（Slice Plot）
        try:
            fig = optuna.visualization.plot_slice(study)
            fig.write_image(f'{output_dir}/slice_plot.png', width=1200, height=800)
            print(f'  ✓ 切片图已保存')
        except Exception as e:
            print(f'  ✗ 切片图生成失败: {e}')
        
        # 5. 等高线图（Contour Plot）- 展示两个最重要超参数的关系
        try:
            # 获取最重要的两个超参数
            importances = optuna.importance.get_param_importances(study)
            top_2_params = list(importances.keys())[:2]
            
            if len(top_2_params) == 2:
                fig = optuna.visualization.plot_contour(study, params=top_2_params)
                fig.write_image(f'{output_dir}/contour_plot.png', width=800, height=600)
                print(f'  ✓ 等高线图已保存 ({top_2_params[0]} vs {top_2_params[1]})')
        except Exception as e:
            print(f'  ✗ 等高线图生成失败: {e}')
        
        # 6. 打印最优参数
        print(f'\n  最优试验 (Trial #{study.best_trial.number}):')
        print(f'    MAE: {study.best_trial.value:.6f}')
        print(f'    参数: {study.best_trial.params}')
        
    except Exception as e:
        print(f'\n{dataset} - 加载 study 失败: {e}')
        print(f'  请确保已运行 tune.py 并生成了数据库文件: {storage}')

def generate_summary_table():
    """
    生成所有数据集的最优参数汇总表
    """
    print('\n' + '='*80)
    print('最优超参数汇总表')
    print('='*80)
    print(f'{"数据集":<10} {"d_model":<10} {"nhead":<8} {"layers":<8} {"dropout":<10} {"lr":<12} {"MAE":<10}')
    print('-'*80)
    
    for dataset in DATASETS:
        study_name = f'{STUDY_NAME_PREFIX}_{dataset}'
        storage = f'sqlite:///{study_name}.db'
        
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            params = study.best_trial.params
            mae = study.best_trial.value
            
            print(f'{dataset:<10} {params["d_model"]:<10} {params["nhead"]:<8} {params["num_layers"]:<8} '
                  f'{params["dropout"]:<10.4f} {params["lr"]:<12.6f} {mae:<10.6f}')
        except:
            print(f'{dataset:<10} {"N/A":<10} {"N/A":<8} {"N/A":<8} {"N/A":<10} {"N/A":<12} {"N/A":<10}')
    
    print('='*80)

if __name__ == '__main__':
    print('='*80)
    print('Optuna 调参可视化工具')
    print('='*80)
    
    # 检查是否安装了 kaleido（用于保存静态图片）
    try:
        import kaleido
        print('✓ kaleido 已安装，可保存静态图片')
    except ImportError:
        print('⚠ kaleido 未安装，请运行: pip install kaleido')
        print('  或使用交互式可视化（不保存图片）')
    
    # 为每个数据集生成可视化
    for dataset in DATASETS:
        visualize_optuna_study(dataset)
    
    # 生成汇总表
    generate_summary_table()
    
    print('\n完成！所有可视化图已保存到 Results/Optuna_{dataset}/ 目录')
    print('\n可用于 PPT 的图片：')
    print('  1. hyperparameter_importance.png - 超参数重要性图（PPT 第10页）')
    print('  2. parallel_coordinate.png - 平行坐标图（PPT 第10页）')
    print('  3. optimization_history.png - 优化历史图（可选）')
    print('  4. slice_plot.png - 切片图（可选）')
    print('  5. contour_plot.png - 等高线图（可选）')
