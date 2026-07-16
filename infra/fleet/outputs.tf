output "workflow_label" {
  description = "Base workflow targeting contract; per-fleet labels are runs-on/fleet=<gpu-linux|gpu-windows>/env=production."
  value       = module.runs_on_fleet.workflow_contract
}

output "stack" {
  description = "Fleet stack metadata."
  value       = module.runs_on_fleet.stack
}

output "config_secret_arn" {
  description = "Rendered Fleet runtime config secret ARN."
  value       = module.runs_on_fleet.config.secret_arn
}
