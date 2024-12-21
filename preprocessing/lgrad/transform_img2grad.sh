#!/bin/bash

GANmodelpath=$(cd $(dirname $0); pwd)
Imgrootdir=$2
Saverootdir=$3
Classes='0_real 1_fake'

# Valdatas='airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'
# Valrootdir=${Imgrootdir}/val/
# Savedir=$Saverootdir/val_grad_pytorch/

# for Valdata in $Valdatas
# do
#     for Class in $Classes
#     do
#         Imgdir=${Valdata}/${Class}
#         CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py\
#             ${Valrootdir}${Imgdir} \
#             ${Savedir}${Imgdir} \
#             ./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth \
#             1
#     done
# done


# Traindatas='airplane bicycle bird boat bottle bus car cat chair cow diningtable dog horse motorbike person pottedplant sheep sofa train tvmonitor'
# Trainrootdir=${Imgrootdir}/train/
# Savedir=$Saverootdir/train_grad_pytorch/
# for Traindata in $Traindatas
# do
#     for Class in $Classes
#     do
#         Imgdir=${Traindata}/${Class}
#         CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py \
#             ${Trainrootdir}${Imgdir} \
#             ${Savedir}${Imgdir} \
#             ./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth \
#             1
#     done
# done


# Testdatas='biggan deepfake gaugan stargan cyclegan/apple cyclegan/horse cyclegan/orangecyclegan/summer cyclegan/winter cyclegan/zebra progan/airplane 
#             progan/bicycle progan/bird progan/boat progan/bottle progan/bus progan/car progan/cat progan/chair progan/cow progan/diningtable progan/dog 
#             progan/horse progan/motorbike progan/person progan/pottedplant progan/sheep progan/sofa progan/train progan/tvmonitor stylegan/bedroom stylegan/car 
#             stylegan/cat stylegan2/car stylegan2/cat stylegan2/church stylegan2/horse crn imle san whichfaceisreal seeingdark'
# Testrootdir=${Imgrootdir}/wang2020/
# Savedir=$Saverootdir/wang2020/

# Testdatas='dalle2 dalle3 firefly glide midjourney-v5 raise stable-diffusion-1-3 stable-diffusion-1-4 stable-diffusion-2 stable-diffusion-xl'
# Testrootdir=${Imgrootdir}/synthbuster/
# Savedir=$Saverootdir/synthbuster/

Testdatas='dalle  glide_100_10  glide_100_27  glide_50_27  guided  imagenet  laion  ldm_100  ldm_200  ldm_200_cfg'
Testrootdir=${Imgrootdir}/ojha2023/
Savedir=$Saverootdir/ojha2023/

for Testdata in $Testdatas
do
    for Class in $Classes
    do
        Imgdir=${Testdata}/${Class}

        # check if the directory exists
        if [ ! -d ${Testrootdir}${Imgdir} ]; then
            continue
        fi

        echo '-----------------------------------------------------------------------------------------------'
        echo ${Testrootdir}${Imgdir}
        echo ${Savedir}${Imgdir}
        CUDA_VISIBLE_DEVICES=$1 python $GANmodelpath/gen_imggrad.py \
            ${Testrootdir}${Imgdir} \
            ${Savedir}${Imgdir} \
            ./weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth \
            256 \
            1 \
            # crop
    done
done

