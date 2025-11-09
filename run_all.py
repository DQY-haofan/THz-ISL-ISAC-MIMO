#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DR-08协议一键运行脚本 (Windows兼容版)
Complete Pipeline Runner for THz-ISAC Analysis

修复内容:
1. 解决Windows GBK编码问题
2. 使用ASCII兼容字符替代Unicode特殊字符
3. 添加编码错误处理

Usage:
    python run_all.py [config.yaml] [options]

Options:
    --skip-threshold    跳过threshold_sweep（耗时较长）
    --skip-supplements  跳过补充图生成
    --quick             快速模式（跳过threshold和supplements）

Author: Generated for DR-08 Protocol
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

# 设置UTF-8编码（Windows兼容）
if sys.platform.startswith('win'):
    try:
        # 尝试设置控制台为UTF-8
        import locale
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'ignore')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'ignore')
    except:
        pass  # 如果失败，使用ASCII替代字符


class Colors:
    """终端颜色（Windows兼容）"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'


def safe_print(text):
    """安全打印，处理编码错误"""
    try:
        print(text)
    except UnicodeEncodeError:
        # 如果编码失败，移除特殊字符
        text = text.replace('✓', '[OK]').replace('✗', '[X]').replace('⚠', '[!]')
        print(text)


def print_header(text):
    """打印标题"""
    safe_print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}")
    safe_print(f"{text:^80}")
    safe_print(f"{'=' * 80}{Colors.END}\n")


def print_step(step_num, total_steps, description):
    """打印步骤"""
    safe_print(f"{Colors.BLUE}{Colors.BOLD}[Step {step_num}/{total_steps}] {description}{Colors.END}")


def print_success(text):
    """打印成功消息"""
    safe_print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_warning(text):
    """打印警告消息"""
    safe_print(f"{Colors.YELLOW}[!] {text}{Colors.END}")


def print_error(text):
    """打印错误消息"""
    safe_print(f"{Colors.RED}[X] {text}{Colors.END}")


def run_command(cmd, description, optional=False):
    """
    运行命令并处理错误

    Args:
        cmd: 要运行的命令（列表形式）
        description: 命令描述
        optional: 是否为可选步骤

    Returns:
        bool: 是否成功
    """
    safe_print(f"\n{Colors.BOLD}Running: {' '.join(cmd)}{Colors.END}")

    start_time = time.time()

    try:
        # Windows兼容的subprocess调用
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'  # 替换无法解码的字符
        )

        elapsed = time.time() - start_time
        print_success(f"{description} completed in {elapsed:.1f}s")

        # 打印输出的最后几行（如果有）
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 5:
                safe_print(f"\n{Colors.BLUE}Last few lines of output:{Colors.END}")
                for line in lines[-5:]:
                    safe_print(f"  {line}")

        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time

        if optional:
            print_warning(f"{description} failed after {elapsed:.1f}s (optional, continuing...)")
            if e.stderr:
                safe_print(f"\nError output:\n{e.stderr}")
            return False
        else:
            print_error(f"{description} failed after {elapsed:.1f}s")
            if e.stderr:
                safe_print(f"\nError output:\n{e.stderr}")
            return False

    except FileNotFoundError:
        print_error(f"Script not found: {cmd[1]}")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return False


def check_files_exist(files):
    """检查必需的文件是否存在"""
    missing = []
    for f in files:
        if not os.path.exists(f):
            missing.append(f)
    return missing


def run_pipeline(config_path, skip_threshold=False, skip_supplements=False):
    """
    运行完整的分析流程

    Args:
        config_path: 配置文件路径
        skip_threshold: 是否跳过threshold_sweep
        skip_supplements: 是否跳过补充图生成

    Returns:
        bool: 是否全部成功
    """

    print_header("DR-08 THz-ISAC ANALYSIS PIPELINE")

    # 检查配置文件
    if not os.path.exists(config_path):
        print_error(f"Configuration file not found: {config_path}")
        return False

    print_success(f"Configuration file: {config_path}")

    # 检查必需的脚本
    required_scripts = [
        'main.py',
        'scan_snr_sweep.py',
        'visualize_results.py'
    ]

    optional_scripts = [
        'threshold_sweep.py',
        'generate_supplementary_figures.py',
        'make_paper_tables.py'
    ]

    missing = check_files_exist(required_scripts)
    if missing:
        print_error(f"Required scripts not found: {missing}")
        return False

    print_success(f"All required scripts found")

    # 检查可选脚本
    available_optional = [s for s in optional_scripts if os.path.exists(s)]
    if len(available_optional) < len(optional_scripts):
        missing_optional = set(optional_scripts) - set(available_optional)
        print_warning(f"Optional scripts not found: {missing_optional}")

    # 创建输出目录
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # 确定总步骤数
    total_steps = 3  # main, snr_sweep, visualize
    if not skip_threshold and 'threshold_sweep.py' in available_optional:
        total_steps += 1
    if not skip_supplements and 'generate_supplementary_figures.py' in available_optional:
        total_steps += 1
    if 'make_paper_tables.py' in available_optional:
        total_steps += 1

    current_step = 0
    failed_steps = []

    # === 步骤1: 主仿真 (必需) ===
    current_step += 1
    print_step(current_step, total_steps, "Main Simulation (Pareto Front Generation)")

    if not run_command(
            ['python', 'main.py', config_path],
            "Main simulation",
            optional=False
    ):
        return False

    # === 步骤2: SNR扫描 (必需) ===
    current_step += 1
    print_step(current_step, total_steps, "SNR Sweep (Capacity Analysis)")

    if not run_command(
            ['python', 'scan_snr_sweep.py', config_path],
            "SNR sweep",
            optional=False
    ):
        failed_steps.append("SNR sweep")
        print_warning("Continuing despite SNR sweep failure...")

    # === 步骤3: Threshold验证 (可选) ===
    if not skip_threshold and 'threshold_sweep.py' in available_optional:
        current_step += 1
        print_step(current_step, total_steps, "Threshold Verification (Whittle vs Cholesky)")

        print_warning("Threshold sweep can be time-consuming. Use --skip-threshold to skip.")

        if not run_command(
                ['python', 'threshold_sweep.py', config_path],
                "Threshold sweep",
                optional=True
        ):
            failed_steps.append("Threshold sweep")

    # === 步骤4: 补充图生成 (推荐) ===
    if not skip_supplements and 'generate_supplementary_figures.py' in available_optional:
        current_step += 1
        print_step(current_step, total_steps, "Supplementary Figures Generation")

        if not run_command(
                ['python', 'generate_supplementary_figures.py', config_path],
                "Supplementary figures",
                optional=True
        ):
            failed_steps.append("Supplementary figures")

    # === 步骤5: 主要图表生成 (必需) ===
    current_step += 1
    print_step(current_step, total_steps, "Main Figures Generation")

    if not run_command(
            ['python', 'visualize_results.py'],
            "Main visualization",
            optional=False
    ):
        failed_steps.append("Main visualization")
        print_warning("Continuing despite visualization failure...")

    # === 步骤6: 表格生成 (可选) ===
    if 'make_paper_tables.py' in available_optional:
        current_step += 1
        print_step(current_step, total_steps, "LaTeX Tables Generation")

        if not run_command(
                ['python', 'make_paper_tables.py'],
                "Table generation",
                optional=True
        ):
            failed_steps.append("Table generation")

    # === 总结 ===
    print_header("PIPELINE EXECUTION SUMMARY")

    # 统计生成的文件
    csv_files = list(Path('results').glob('*.csv'))
    pdf_files = list(Path('figures').glob('*.pdf'))
    png_files = list(Path('figures').glob('*.png'))

    safe_print(f"\n{Colors.BOLD}Generated Files:{Colors.END}")
    safe_print(f"  CSV data files: {len(csv_files)}")
    safe_print(f"  PDF figures: {len(pdf_files)}")
    safe_print(f"  PNG figures: {len(png_files)}")

    if csv_files:
        safe_print(f"\n{Colors.BOLD}Key CSV Files:{Colors.END}")
        for f in sorted(csv_files)[:5]:
            size_kb = f.stat().st_size / 1024
            safe_print(f"  [OK] {f.name} ({size_kb:.1f} KB)")
        if len(csv_files) > 5:
            safe_print(f"  ... and {len(csv_files) - 5} more")

    if pdf_files:
        safe_print(f"\n{Colors.BOLD}Key PDF Figures:{Colors.END}")
        for f in sorted(pdf_files)[:5]:
            safe_print(f"  [OK] {f.name}")
        if len(pdf_files) > 5:
            safe_print(f"  ... and {len(pdf_files) - 5} more")

    # 失败步骤总结
    if failed_steps:
        safe_print(f"\n{Colors.YELLOW}{Colors.BOLD}Failed/Skipped Steps:{Colors.END}")
        for step in failed_steps:
            safe_print(f"  [!] {step}")
        safe_print(f"\n{Colors.YELLOW}Pipeline completed with some warnings{Colors.END}")
    else:
        safe_print(f"\n{Colors.GREEN}{Colors.BOLD}[OK] All steps completed successfully!{Colors.END}")

    safe_print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    safe_print(f"  1. Review figures in: ./figures/")
    safe_print(f"  2. Review data in: ./results/")
    safe_print(f"  3. Check paper tables if generated")

    return len(failed_steps) == 0


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description='DR-08 Protocol Complete Pipeline Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 运行完整流程
  python run_all.py config.yaml

  # 快速模式（跳过耗时步骤）
  python run_all.py config.yaml --quick

  # 跳过threshold验证
  python run_all.py config.yaml --skip-threshold

  # 跳过补充图
  python run_all.py config.yaml --skip-supplements
        """
    )

    parser.add_argument(
        'config',
        nargs='?',
        default='config.yaml',
        help='Configuration file path (default: config.yaml)'
    )

    parser.add_argument(
        '--skip-threshold',
        action='store_true',
        help='Skip threshold_sweep.py (time-consuming)'
    )

    parser.add_argument(
        '--skip-supplements',
        action='store_true',
        help='Skip generate_supplementary_figures.py'
    )

    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode (skip threshold and supplements)'
    )

    args = parser.parse_args()

    # Quick mode implies skipping both
    if args.quick:
        args.skip_threshold = True
        args.skip_supplements = True

    # 记录开始时间
    start_time = time.time()

    # 运行流程
    success = run_pipeline(
        args.config,
        skip_threshold=args.skip_threshold,
        skip_supplements=args.skip_supplements
    )

    # 总耗时
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)

    safe_print(f"\n{Colors.BOLD}Total execution time: {minutes}m {seconds}s{Colors.END}")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())