from django.db import models
from datetime import datetime


# Create your models here.
class ClassAttention(models.Model):
    hash_key = models.ForeignKey('ClassAttentionID', on_delete=models.CASCADE)
    time_stamp = models.DateTimeField(default=datetime.now())
    gz_attn = models.CharField(max_length=100)
    ps_attn = models.CharField(max_length=100)
    sleep_n = models.CharField(max_length=100)
    ov_attn = models.CharField(max_length=100)
    n_q = models.CharField(max_length=100)
    n_b = models.CharField(max_length=100)
    n_p = models.CharField(max_length=100)
    pos_attn = models.CharField(max_length=100)
    frame_counter = models.IntegerField(default=-1)
    time_elapsed = models.DecimalField(null=True, max_digits=6, decimal_places=2)

    def __str__(self):
        return (self.hash_key.class_id + str(self.time_stamp))


class ClassAttentionID(models.Model):
    hash_key = models.CharField(max_length=100)
    session_teacher = models.CharField(max_length=100)
    time_stamp = models.DateTimeField(auto_now_add=True, null=True)
    class_id = models.CharField(max_length=100, unique=True)


class ClassAttendance(models.Model):
    session_teacher = models.CharField(max_length=100)
    median_attn = models.CharField(max_length=200)
    num_students = models.CharField(max_length=200)

