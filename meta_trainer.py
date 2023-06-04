import torch
from apex import amp
import torch.nn.functional as F
from utils.misc import AverageMeter
import pdb


def meta_trainer(args, model, criterion, optimizer, train_loader, device, tips):
    e, n_iter_train, max_norm, amp_opt_level, lr = tips
    losses = AverageMeter()
    train_acc = []
    model.train()

    if args.method == "hcl":
        for idx, (imgs, vids, labels) in enumerate(train_loader):   
            imgs = imgs.to(device)  # 48, T, C, H, W
            labels = labels.to(device)
            bsz = labels.shape[0]
            
            logits, contrasts = model(imgs, vids)  # 

            acc = (logits.argmax(1)==labels).type(torch.cuda.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            loss, ce_loss, contrast_loss = criterion(contrasts, logits, labels)
            losses.update(loss.item(), bsz)
       
            optimizer.zero_grad()
            if amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
         
            if idx % args.print_iter == 0:
                print('Train: [epoch: %d][iter: %d][lr: %.6f][iter: %d][loss: %.4f][ce: %.3f][global_cts: %.3f][temp_cts: %.3f][temp_cycle: %.3f][spa_cts: %.3f][spa_cycle: %.3f][avg_loss: %.3f][acc: %.3f]'%(
                    e, idx, lr, idx, losses.val, ce_loss.item(), contrast_loss["global"].item(), 
                    contrast_loss["temp"].item(), contrast_loss["temp_cycle"].item(), 
                    contrast_loss["spatial"].item(), contrast_loss["spa_cycle"].item(),
                    losses.avg, acc))

            n_iter_train += 1
            break
    else:  
        # Meta Train
        print("Train... {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
        train_acc, train_loss = [], []
  
        for i, (imgs, _, _) in enumerate(train_loader):
            imgs = imgs.to(device)  # way*(shot+query), t, c, h, w
            
            support_labels = torch.arange(args.way).reshape(args.way, 1).repeat(1, args.shot).to(device).reshape(-1)
            query_labels = torch.arange(args.way).reshape(args.way, 1).repeat(1, args.query).to(device).reshape(-1)
            labels = torch.cat([support_labels, query_labels])
 
            logits, preds = model(imgs, labels) 
            loss = F.cross_entropy(logits, query_labels)
            loss = loss / args.episode_per_batch
            
            if amp_opt_level != "O0":
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
           
            train_loss.append(loss.item())
            total_loss = sum(train_loss)/len(train_loss)

            # update weight every 16 episodes
            if (i+1) % args.episode_per_batch == 0:
                optimizer.step()
                optimizer.zero_grad()

            # calculate accuracy
            if args.method == 'relation':
                logits, _ = logits.reshape(args.way*args.query, args.way, args.shot).max(-1)
   
            acc = (preds.view(-1) == query_labels).type(torch.cuda.FloatTensor).mean().item()
            train_acc.append(acc)
            total_acc = sum(train_acc) / len(train_acc)

            if i % args.print_iter == 0:
                print('Train: [epoch: %d][iter: %d][loss: %.4f (mean: %.4f)][acc: %.4f (mean: %.4f)]'%(
                       e, i, loss.item()*args.episode_per_batch, total_loss*args.episode_per_batch, acc * 100, total_acc * 100))
                
            n_iter_train += 1

    return n_iter_train
            
    
    
