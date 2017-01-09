obj-m := get_idt_info.o
KDIR := /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

default:
	$(MAKE) -C $(KDIR) SUBDIRS=$(PWD) modules

insmod:
	sudo insmod get_idt_info.ko

rmmod:
	sudo rmmod get_idt_info.ko

reload:
	-make rmmod
	make insmod

clean:
	rm *.o
	rm *.ko
	rm *.mod.c
