#include <linux/kernel.h>
#include <linux/module.h>

MODULE_AUTHOR("up");
MODULE_DESCRIPTION("Anonymous inode mapping to userspace");
MODULE_LICENSE("GPL");

struct {
	uint16_t limit;
	void *base;
} __attribute__ ((packed)) idtr;

unsigned long long get_handler_addr(struct gate_struct64 *p)
{
	return ((unsigned long long)p->offset_high << 32) |
		((unsigned long long)p->offset_middle << 16) |
		(unsigned long long)p->offset_low;
}

static int __init get_idt_info_init(void)
{
	struct gate_struct64 *ptr;
	int i;

	printk(KERN_INFO "get_idt_info: init\n");

	asm("sidt %0" : "=m" (idtr));
	ptr = idtr.base;
	printk(KERN_INFO "get_idt_info: base_addr = 0x%p\n", ptr);

	for (i = 0; i < (idtr.limit / sizeof(struct gate_struct64)) + 1; i++) {
		printk(KERN_INFO "get_idt_info: %03d 0x%llx\n", i, get_handler_addr(&ptr[i]));
	}

	return 0;
}
module_init(get_idt_info_init);

static void __exit get_idt_info_exit(void)
{
	printk(KERN_INFO "get_idt_info: exit\n");
}
module_exit(get_idt_info_exit);


				  
