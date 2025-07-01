import argparse
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_error_detection_failure(json_file_path: str) -> Tuple[int, int, float]:
    """
    Analyze the error detection failure rate from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing analysis results
        
    Returns:
        Tuple of (total_cases, failed_detections, failure_rate)
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file: {e}")
        return 0, 0, 0.0
    
    # Extract files_analyzed from the JSON structure
    files_analyzed = data.get('files_analyzed', [])
    
    if not files_analyzed:
        print("No files analyzed found in the JSON file.")
        return 0, 0, 0.0
    
    total_cases = len(files_analyzed)
    failed_detections = 0
    
    # Count cases where error_detected is false
    for file_data in files_analyzed:
        analysis_result = file_data.get('analysis_result', {})
        error_detected = analysis_result.get('error_detected', True)
        
        if not error_detected:
            failed_detections += 1
    
    failure_rate = failed_detections / total_cases if total_cases > 0 else 0.0
    
    return total_cases, failed_detections, failure_rate


def generate_visualization(total_cases: int, failed_detections: int, failure_rate: float, output_dir: str):
    """
    Generate visualization charts for error detection analysis.
    
    Args:
        total_cases: Total number of cases analyzed
        failed_detections: Number of cases where error detection failed
        failure_rate: Error detection failure rate
        output_dir: Directory to save the visualization
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart for error detection results
    successful_detections = total_cases - failed_detections
    labels = ['Successful Detection', 'Failed Detection']
    sizes = [successful_detections, failed_detections]
    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)  # explode the failed detection slice
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.set_title('Error Detection Results Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart for detailed statistics
    categories = ['Total Cases', 'Successful\nDetections', 'Failed\nDetections']
    values = [total_cases, successful_detections, failed_detections]
    bar_colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = ax2.bar(categories, values, color=bar_colors, alpha=0.8)
    ax2.set_title('Error Detection Statistics', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Cases', fontsize=12)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold')
    
    # Add failure rate text
    ax2.text(0.5, 0.95, f'Failure Rate: {failure_rate:.2%}', 
             transform=ax2.transAxes, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'error_detection_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def print_detailed_analysis(json_file_path: str, total_cases: int, failed_detections: int, failure_rate: float):
    """
    Print detailed analysis results to console.
    
    Args:
        json_file_path: Path to the analyzed JSON file
        total_cases: Total number of cases
        failed_detections: Number of failed detections
        failure_rate: Failure rate
    """
    successful_detections = total_cases - failed_detections
    
    print("=" * 80)
    print("ERROR DETECTION FAILURE ANALYSIS RESULTS")
    print("=" * 80)
    print(f"Input File: {json_file_path}")
    print(f"Analysis Date: {Path(json_file_path).stat().st_mtime}")
    print("-" * 80)
    print(f"Total Cases Analyzed: {total_cases}")
    print(f"Successful Error Detections: {successful_detections}")
    print(f"Failed Error Detections: {failed_detections}")
    print("-" * 80)
    print(f"Error Detection Failure Rate: {failure_rate:.2%} ({failure_rate:.4f})")
    print(f"Error Detection Success Rate: {(1-failure_rate):.2%} ({(1-failure_rate):.4f})")
    print("=" * 80)
    
    if failure_rate > 0.5:
        print("⚠️  WARNING: High failure rate detected! More than 50% of errors were not detected.")
    elif failure_rate > 0.3:
        print("⚠️  CAUTION: Moderate failure rate. Consider improving error detection mechanisms.")
    else:
        print("✅ GOOD: Low failure rate. Error detection is performing well.")
    
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze error detection failure rate from failure attribution JSON results."
    )
    
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the JSON file containing failure attribution analysis results."
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Directory to save visualization charts (default: current directory)."
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.json_file):
        print(f"Error: JSON file '{args.json_file}' does not exist.")
        return
    
    if not args.json_file.endswith('.json'):
        print(f"Error: Input file must be a JSON file.")
        return
    
    print(f"Analyzing error detection failure rate from: {args.json_file}")
    print("-" * 80)
    
    # Analyze the JSON file
    total_cases, failed_detections, failure_rate = analyze_error_detection_failure(args.json_file)
    
    if total_cases == 0:
        print("No valid data found for analysis.")
        return
    
    # Print detailed analysis
    print_detailed_analysis(args.json_file, total_cases, failed_detections, failure_rate)
    
    # Generate visualization
    try:
        generate_visualization(total_cases, failed_detections, failure_rate, args.output_dir)
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Visualization chart saved in: {args.output_dir}")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("Analysis completed, but visualization could not be generated.")


if __name__ == "__main__":
    main()