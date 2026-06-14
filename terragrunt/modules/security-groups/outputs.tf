output "alb_sg_id" {
  value = aws_security_group.alb.id
}

output "app_sg_id" {
  value = aws_security_group.app.id
}

output "rds_sg_id" {
  value = aws_security_group.data["rds"].id
}

output "redis_sg_id" {
  value = aws_security_group.data["redis"].id
}

output "efs_sg_id" {
  value = aws_security_group.data["efs"].id
}
