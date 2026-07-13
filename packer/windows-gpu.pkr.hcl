packer {
  required_plugins {
    amazon = {
      source  = "github.com/hashicorp/amazon"
      version = ">= 1.8.0, < 2.0.0"
    }
  }
}

# Region MUST match the region your RunsOn CloudFormation stack deploys
# runners into (custom AMIs are region-local).
variable "region" {
  type    = string
  default = "ap-southeast-2"
}

# Builder needs a physical GPU present so the driver binds during the
# bake. Baked on the cheapest T4 box; the AWS GRID driver it installs is
# multi-GPU, so the resulting AMI also runs on G5 (A10G) and G6 (L4).
# Requested as spot (see spot_price) because this region has no On-Demand
# G quota. Kept as a single instance_type + spot_price ("simple" spot
# request) rather than spot_instance_types: the latter uses the EC2 Fleet
# API, which needs ec2:DescribeInstanceTypeOfferings on the builder role.
variable "instance_type" {
  type    = string
  default = "g4dn.xlarge"
}

# Max hourly spot bid. AWS never charges above the on-demand price
# (g4dn.xlarge is ~$0.75/h in ap-southeast-2), so this ceiling only has to
# clear the market spot price; you pay the actual (lower) spot rate.
variable "spot_price" {
  type    = string
  default = "1.00"
}

# Empty -> Packer picks a subnet in the default VPC. Set explicitly if the
# account has no default VPC.
variable "subnet_id" {
  type    = string
  default = ""
}

# AWS account that publishes the (public) RunsOn base AMIs.
variable "runs_on_owner" {
  type    = string
  default = "135269210855"
}

# RunsOn Windows 2025 base image; already carries the RunsOn agent, so a
# 14-day rebuild keeps the agent under GitHub's 30-day dispatch cutoff.
variable "source_ami_name" {
  type    = string
  default = "runs-on-v2.2-windows25-full-x64-*"
}

locals {
  timestamp = formatdate("YYYYMMDD-hhmmss", timestamp())
}

source "amazon-ebs" "windows_gpu" {
  region                                     = var.region
  instance_type                              = var.instance_type
  spot_price                                 = var.spot_price
  subnet_id                                  = var.subnet_id
  associate_public_ip_address                = true
  temporary_security_group_source_public_ip  = true
  ebs_optimized                              = true
  force_deregister                           = true
  force_delete_snapshot                      = true

  communicator   = "winrm"
  winrm_username = "Administrator"
  winrm_use_ssl  = true
  winrm_insecure = true
  winrm_timeout  = "30m"

  aws_polling {
    delay_seconds = 30
    max_attempts  = 300
  }

  source_ami_filter {
    filters = {
      name                = var.source_ami_name
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    owners      = [var.runs_on_owner]
    most_recent = true
  }

  # EC2Launch user-data enabling WinRM over HTTPS so Packer can connect.
  # Bootstrap pattern from runs-on/runner-images-for-aws
  # (patches/windows/templates/windows25-gpu-x64.pkr.hcl).
  user_data = <<EOF
<powershell>
Enable-PSRemoting -SkipNetworkProfileCheck -Force
winrm set winrm/config/service/auth '@{Basic="true"}'
Set-Service -Name WinRM -StartupType Automatic
$Cert = New-SelfSignedCertificate -CertstoreLocation Cert:\LocalMachine\My -DnsName "cubie-packer"
Get-ChildItem WSMan:\Localhost\Listener | Where-Object Keys -eq "Transport=HTTP" | Remove-Item -Recurse
New-Item -Path WSMan:\LocalHost\Listener -Transport HTTPS -Address * -CertificateThumbPrint $Cert.Thumbprint -Force
New-NetFirewallRule -DisplayName "Windows Remote Management (HTTPS-In)" -Name "Windows Remote Management (HTTPS-In)" -Profile Any -LocalPort 5986 -Protocol TCP
</powershell>
<persist>false</persist>
EOF

  ami_name        = "cubie-win-gpu-${local.timestamp}"
  ami_description = "RunsOn Windows 2025 + NVIDIA T4 GRID driver for cubie CUDA CI"

  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 100
    volume_type           = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name      = "cubie-win-gpu-${local.timestamp}"
    Project   = "cubie"
    Purpose   = "cuda-ci-windows-gpu"
    BaseImage = var.source_ami_name
  }
}

build {
  sources = ["source.amazon-ebs.windows_gpu"]

  # Install the AWS GRID driver, restart, then verify (marker pattern in
  # the script short-circuits the second run to nvidia-smi verification).
  provisioner "powershell" {
    pause_before = "2m0s"
    scripts      = ["ci/tools/install_gpu_driver.ps1"]
  }

  provisioner "windows-restart" {
    restart_timeout = "30m"
  }

  provisioner "powershell" {
    scripts = ["ci/tools/install_gpu_driver.ps1"]
  }

  # Disable WinRM in the published AMI and reset EC2Launch so it
  # re-initialises (and re-runs the RunsOn agent bootstrap) on first boot as
  # a runner. Block mirrors runs-on/runner-images-for-aws
  # (windows25-gpu-x64.pkr.hcl) so capture does not need a WinRM reconnect.
  provisioner "powershell" {
    inline = [
      "Write-Host 'Disabling WinRM in the published AMI...'",
      "Set-Service -Name WinRM -StartupType Disabled",
      "Write-Host 'Scheduling WinRM shutdown so Packer does not need to reconnect after final capture starts...'",
      "$null = Start-Process -FilePath 'powershell.exe' -WindowStyle Hidden -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', \"Start-Sleep -Seconds 15; Stop-Service -Name WinRM -Force -ErrorAction SilentlyContinue\")",
      "$OSVersion = [System.Environment]::OSVersion.Version",
      "if ($OSVersion.Major -eq 10 -and $OSVersion.Build -ge 20348) {",
      "    Write-Host 'Windows Server 2022+ detected, using EC2Launch v2'",
      "    & \"C:\\Program Files\\Amazon\\EC2Launch\\EC2Launch.exe\" reset",
      "} else {",
      "    Write-Host 'Windows Server pre-2022 detected, using EC2Launch v1'",
      "    & C:\\ProgramData\\Amazon\\EC2-Windows\\Launch\\Scripts\\InitializeInstance.ps1 -Schedule",
      "}",
      "exit 0",
    ]
  }
}
