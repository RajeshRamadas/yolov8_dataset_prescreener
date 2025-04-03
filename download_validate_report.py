#!/usr/bin/env python3
import os
import sys
import subprocess
import gdown
import zipfile
import smtplib
import argparse
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Download dataset, validate, and send report')
    parser.add_argument('--drive_url', type=str, default='https://drive.google.com/file/d/1R44tNwMYBU3kaQLB2cgqzb8HMNisEuqA/view?usp=sharing',
                        help='Google Drive URL of the dataset')
    parser.add_argument('--email', type=str, required=True,
                        help='Email to send the report to')
    parser.add_argument('--issue_threshold', type=int, default=10,
                        help='Threshold for number of issues')
    return parser.parse_args()

def download_from_gdrive(url, output_path='dataset.zip'):
    """Download file from Google Drive URL"""
    print(f"Downloading dataset from {url}...")
    gdown.download(url=url, output=output_path, quiet=False, fuzzy=True)
    return os.path.abspath(output_path)

def extract_dataset(zip_path, extract_dir='dataset'):
    """Extract the dataset from zip file"""
    print(f"Extracting dataset to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    return os.path.abspath(extract_dir)

def find_yaml_file(dataset_dir):
    """Find the data.yaml file in the dataset directory"""
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file == 'data.yaml':
                yaml_path = os.path.join(root, file)
                print(f"Found data.yaml at: {yaml_path}")
                return yaml_path
    
    print("WARNING: data.yaml not found in the dataset directory")
    return None

def run_validation(dataset_dir, yaml_path, output_dir='validation_results', issue_threshold=10):
    """Run the YOLOv8 dataset validator"""
    print(f"Running dataset validation...")
    
    # Create timestamp for output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"{output_dir}_{timestamp}"
    
    # Build the command
    cmd = [
        'python', 'data_validator.py',
        '--dataset_path', dataset_dir,
        '--output_dir', output_dir,
        '--json_report',
        '--fail_on_issues',
        '--issue_threshold', str(issue_threshold)
    ]
    
    # Add YAML path if found
    if yaml_path:
        cmd.extend(['--yaml_path', yaml_path])
    
    # Run the validator
    try:
        subprocess.run(cmd, check=True)
        print(f"Validation completed successfully. Results in {output_dir}")
        return output_dir, True
    except subprocess.CalledProcessError as e:
        print(f"Validation failed with error: {e}")
        return output_dir, False

def send_email_report(email_to, output_dir, yaml_path, success):
    """Send validation report via email"""
    print(f"Sending email report to {email_to}...")
    
    # Set up email content
    msg = MIMEMultipart()
    msg['From'] = 'raksbangs@gmail.com'  # Replace with your email
    msg['To'] = email_to
    msg['Subject'] = f"YOLOv8 Dataset Validation Report - {'SUCCESS' if success else 'ISSUES FOUND'}"
    
    # Email body
    body = f"""
    YOLOv8 Dataset Validation Report
    ===============================
    
    Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    Status: {'SUCCESS' if success else 'ISSUES FOUND'}
    Dataset YAML File: {yaml_path if yaml_path else 'Not found'}
    
    Please see the attached report summary and check the full HTML report
    in the validation output directory: {output_dir}
    """
    msg.attach(MIMEText(body, 'plain'))
    
    # Attach summary file if it exists
    summary_file = os.path.join(output_dir, "summary_report.txt")
    if os.path.exists(summary_file):
        with open(summary_file, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype='txt')
            attachment.add_header('Content-Disposition', 'attachment', filename='summary_report.txt')
            msg.attach(attachment)
    
    # Attach issues file if it exists
    issues_file = os.path.join(output_dir, "issues.csv")
    if os.path.exists(issues_file):
        with open(issues_file, 'rb') as f:
            attachment = MIMEApplication(f.read(), _subtype='csv')
            attachment.add_header('Content-Disposition', 'attachment', filename='issues.csv')
            msg.attach(attachment)
    
    # For demonstration purposes - in a real scenario, you would configure SMTP properly
    print(f"EMAIL WOULD BE SENT TO: {email_to}")
    print(f"EMAIL SUBJECT: {msg['Subject']}")
    print(f"EMAIL BODY:\n{body}")
    print("ATTACHMENTS: summary_report.txt, issues.csv (if available)")
    
    # Uncomment these lines to actually send the email via SMTP
    """
    with smtplib.SMTP('smtp.yourcompany.com', 587) as server:
        server.starttls()
        server.login('your_username', 'your_password')
        server.send_message(msg)
    """

def main():
    # Parse arguments
    args = parse_args()
    
    # Create a working directory
    base_dir = os.path.abspath(os.path.dirname(__file__))
    os.chdir(base_dir)
    
    # Download the dataset
    zip_path = download_from_gdrive(args.drive_url)
    
    # Extract the dataset
    dataset_dir = extract_dataset(zip_path)
    
    # Find the data.yaml file
    yaml_path = find_yaml_file(dataset_dir)
    
    # Run validation
    output_dir, success = run_validation(dataset_dir, yaml_path, issue_threshold=args.issue_threshold)
    
    # Send email report
    send_email_report(args.email, output_dir, yaml_path, success)
    
    # Clean up (optional)
    # os.remove(zip_path)
    
    print("Process completed!")
    
    # Return success status
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())