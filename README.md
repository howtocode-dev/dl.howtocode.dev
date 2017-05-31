# ডিপ লার্নিং ও আর্টিফিশিয়াল নিউরাল নেটওয়ার্ক

<iframe src="https://www.facebook.com/plugins/like.php?href=https%3A%2F%2Fwww.facebook.com%2Fhowtocode.com.bd%2F&width=450&layout=standard&action=like&size=small&show_faces=true&share=true&height=80&appId=353725671441956" width="450" height="80" style="border:none;overflow:hidden" scrolling="no" frameborder="0" allowTransparency="true"></iframe>  

**কোর্স পরিচালনায়**  
[Nuhil Mehdy](https://nuhil.net)   

**স্বয়ংক্রিয় কন্ট্রিবিউটরের তালিকা**  
(প্রথম ৫ জন)  
<iframe scrolling="auto" frameborder="0" style="border:none; overflow:hidden; height:115px; width:100%; margin-left: 15;" allowTransparency="true" src="https://nuhil.github.io/api/contributions.html?repo=dl"></iframe>

**ভূমিকা**   
দেরি করে হলেও ডিপ লার্নিং এর ব্যবহার ও প্রয়োজনীয়তা ইদানীং ব্যাপক হারে বাড়ছে। কম্পিউটার ভিশন, ন্যাচারাল ল্যাঙ্গুয়েজ প্রসেসিং সহ বেশ কিছু সেক্টরে এর প্রভাব লক্ষণীয়। ডিপ লার্নিং হচ্ছে মেশিন লার্নিং এর একটি ব্র্যাঞ্চ বা একটা মেশিন লার্নিং টেকনিক যা কিনা নিজে নিজেই সরাসরি ডাটা থেকে ফিচার এবং টাস্ক শিখে নিতে পারে। সেই ডাটা হতে পারে ইমেজ, টেক্সট এমনকি সাউন্ড। অনেকেই ডিপ লার্নিং -কে এন্ড টু এন্ড লার্নিং-ও বলে থাকেন। ডিপ লার্নিং টেকনিকের খুব পুরনো এবং বহুল পরিচিত ব্যাবহার হয় পোস্টাল সার্ভিসে খামের উপর বিভিন্ন ধরনের হাতের লেখা চিহ্নিত করতে। মোটামুটি ১৯৯০ সালের দিক থেকেই ডিপ লার্নিং এর এই প্রয়োগ চলে আসছে।  

২০০৪/২০০৫ সালের দিক থেকে ডিপ লার্নিং এর ব্যবহার খুব উল্লেখ যোগ্য ভাবে বেড়ে চলছে। মূলত তিনটি কারণে — প্রথমত ইদানিং কালের ডিপ লার্নিং মেথড গুলো মানুষের চেয়ে অনেক বেশি ভালো ভাবে অবজেক্ট রিকগনিশনের বা ক্লাসিফিকেশনের কাজ করতে পারছে, দ্বিতীয়ত GPU এর কল্যাণে অনেক বড় আকারের ডিপ নেটওয়ার্ক খুব কম সময়ের মধ্যেই লার্নিং শেষ করে নিতে পারছে, তৃতীয়ত, খুব ইফেক্টিভ লার্নিং এর জন্য যে পরিমাণ ডাটার প্রয়োজন পরে সেই লেভেলের ডাটা গত ৫/৬ বছরে ব্যবহার উপযোগীভাবে তৈরি হচ্ছে বিভিন্ন সার্ভিসের মাধ্যমে।  

বেশির ভাগ ডিপ লার্নিং মেথড নিউরাল নেটওয়ার্ক আর্কিটেকচার ফলো করে আর তাই ডিপ লার্নিং মডেলকে মাঝে মধ্যেই ডিপ নিউরাল নেটওয়ার্ক হিসেবেও বলা হয়ে থাকে। খুব পপুলার একটি ডিপ লার্নিং মডেল হচ্ছে কনভলিউশনাল নিউরাল নেটওয়ার্ক বা CNN. এ ধরনের নেটওয়ার্ক বিশেষ করে ইমেজ ডাটা নিয়ে কাজ করার সময় ব্যবহৃত হয়ে থাকে। যখন বেশ কিছু লেয়ার নিয়ে একটি নিউরাল নেটওয়ার্ক ডিজাইন করা হয় তখন এটাকেই ডীপ নিউরাল নেটওয়ার্ক বলে। এই লেয়ারের সংখ্যা হতে পারে ২-৩ টি থেকে শ-খানেক পর্যন্ত।  

এ পর্যন্ত পড়ার পর যদি খুব অস্বস্তি চলে আসে তবে ভয় পাওয়ার কিছু নাই, নিচেই খুব ব্যাসিক কিছু উদাহরণ এর মাধ্যমে সব সহজ ভাবে আলোচনা করা হবে। আমরা একটা সমস্যা দেখবো এবং তার সমাধানের জন্য একটি নিউরাল নেটওয়ার্ক ডিজাইন করবো। তারপর পাইথনে কোড লিখে সেই নেটওয়ার্কের প্রোগ্রামেটিক ভার্শন লিখবো এবং সেটার লার্নিং করিয়ে সমস্যাটা সমাধানও করবো ইনসা আল্লাহ। তার আগে পরবর্তী চ্যাপ্টারে জেনে নেব, মেশিন লার্নিং এবং ডিপ লার্নিং এর মধ্যে পার্থক্য বা সম্পর্ক কোথায়।  

**ওপেন সোর্স**

এই বইটি মূলত স্বেচ্ছাশ্রমে লেখা এবং বইটি সম্পূর্ন ওপেন সোর্স । এখানে তাই আপনিও অবদান রাখতে পারেন লেখক হিসেবে । আপনার কন্ট্রিবিউশান গৃহীত হলে অবদানকারীদের তালিকায় আপনার নাম স্বয়ংক্রিয়ভাবে যুক্ত হয়ে যাবে।  

এটি মূলত একটি [গিটহাব রিপোজিটোরি](https://github.com/howtocode-com-bd/dl.howtocode.com.bd)  যেখানে এই বইয়ের আর্টিকেল গুলো মার্কডাউন ফরম্যাটে লেখা হচ্ছে । রিপোজিটরিটি ফর্ক করে পুল রিকুয়েস্ট পাঠানোর মাধ্যমে আপনারাও অবদান রাখতে পারেন । বিস্তারিত দেখতে পারেন এই ভিডিওতে  [Video](http://blog.howtocode.com.bd/?p=32)

> **বর্তমানে বইটির কন্টেন্ট বিভিন্ন কন্ট্রিবিউটর এবং নানা রকম সোর্স থেকে সংগৃহীত এবং সংকলিত।**

<iframe src="https://www.facebook.com/plugins/like.php?href=http%3A%2F%2Fdl.howtocode.com.bd&amp;width&amp;layout=button_count&amp;action=like&amp;show_faces=false&amp;share=true&amp;height=21&amp;appId=353725671441956" scrolling="no" frameborder="0" style="border:none; overflow:hidden; height:21px;" allowTransparency="true"></iframe>   

[![Join the chat at https://gitter.im/howtocode-com-bd/dl.howtocode.com.bd](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/howtocode-com-bd/dl.howtocode.com.bd?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

<a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-nd/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-nd/4.0/">Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License</a>.