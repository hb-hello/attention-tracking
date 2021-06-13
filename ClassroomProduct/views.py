from ClassApp import forms
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from ClassApp.models import *
# from qr_code.qrcode.utils import QRCodeOptions
import urllib.parse
from django.utils import timezone
from datetime import datetime


@login_required
def home(request):
    # request.session.pop('class_id')
    # request.session.pop('start_time')
    # request.session.modified=True
    return render(request, 'home.html')


def signout(request):
    return render(request, 'registration/logout.html')


def record_attendance(request):
    return render(request, 'attendance/record_attendance.html')


@login_required
def create_session(request):
    if request.method == 'POST':
        form = forms.CreateClassAttentionID(request.POST)
        if form.is_valid():
            instance = form.save(commit=False)
            instance.session_teacher = request.user
            instance.hash_key = urllib.parse.quote("{0}_{1}".format(request.user, request.POST.get('class_id')))
            instance.save()
            request.session['class_id'] = instance.class_id
            request.session['start_time'] = str(datetime.now())
            request.session.modified=True
            return redirect('streamsession')
    else:
        form = forms.CreateClassAttentionID()
    return render(request, 'attention/create_session.html', {'form': form})


# def qr_code_session(request):
#     prim_key = list(ClassAttentionID.objects.all().filter(session_teacher="rebecca"))[-1].hash_key
#     context = dict(key=prim_key, my_options=QRCodeOptions(size='S', border=6, error_correction='L', image_format='png'))
#     return render(request, 'attention/qr_code_session.html', context=context)


def stream_session(request):
    return render(request, 'attention/stream_session.html')


def get_list_session(request):
    sessions_list = ClassAttentionID.objects.filter(session_teacher=request.user)
    return render(request, 'analytics/session_list.html', {'sessions_list': sessions_list})

def get_attendance(request):
    return render(request, 'attendance/record_attendance.html')

# def get_session_analytics(request, parameter1):
#     hk = parameter1
#     print("function hit")
#     return render(request, 'analytics/session_analytics.html', {"hk": hk})

def get_session_analytics(request):
    return render(request, 'analytics/session_analytics.html')
