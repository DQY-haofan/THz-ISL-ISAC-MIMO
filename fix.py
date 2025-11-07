#!/usr/bin/env python3
"""
配置文件自动修复工具
自动将字符串类型的数值转换为正确的类型
"""

import yaml
import re
import sys


def fix_config(input_path='config.yaml', output_path='config_fixed.yaml'):
    """自动修复配置文件中的类型问题"""

    print("=" * 80)
    print("配置文件自动修复工具")
    print("=" * 80)

    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 成功加载配置文件: {input_path}\n")
    except Exception as e:
        print(f"✗ 加载配置文件失败: {e}")
        return False

    def convert_value(value):
        """尝试将字符串转换为数值"""
        if not isinstance(value, str):
            return value

        # 尝试转换为整数
        try:
            return int(value)
        except ValueError:
            pass

        # 尝试转换为浮点数
        try:
            return float(value)
        except ValueError:
            pass

        # 科学计数法
        if re.match(r'^-?\d+\.?\d*[eE][+-]?\d+$', value):
            try:
                return float(value)
            except ValueError:
                pass

        # 保持原样
        return value

    def fix_dict(d):
        """递归修复字典中的所有值"""
        if not isinstance(d, dict):
            return d

        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = fix_dict(value)
            elif isinstance(value, list):
                d[key] = [convert_value(v) if not isinstance(v, dict) else fix_dict(v) for v in value]
            else:
                d[key] = convert_value(value)

        return d

    # 修复配置
    config_fixed = fix_dict(config)

    # 显示修复的项
    print("修复的项目:")
    changes = []

    def find_changes(orig, fixed, path=''):
        if isinstance(orig, dict) and isinstance(fixed, dict):
            for key in orig:
                new_path = f"{path}.{key}" if path else key
                if key in fixed:
                    find_changes(orig[key], fixed[key], new_path)
        elif isinstance(orig, str) and not isinstance(fixed, str):
            changes.append(f"  {path}: '{orig}' → {fixed} ({type(fixed).__name__})")

    find_changes(config, config_fixed)

    if changes:
        for change in changes:
            print(change)
        print(f"\n共修复 {len(changes)} 个项目")
    else:
        print("  无需修复，配置文件已经正确")

    # 保存修复后的配置
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_fixed, f, default_flow_style=False, sort_keys=False)
        print(f"\n✓ 修复后的配置已保存到: {output_path}")
        print(f"\n使用方法：")
        print(f"  python scan_snr_sweep.py {output_path}")
        return True
    except Exception as e:
        print(f"\n✗ 保存失败: {e}")
        return False


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'config.yaml'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'config_fixed.yaml'

    success = fix_config(input_file, output_file)
    sys.exit(0 if success else 1)