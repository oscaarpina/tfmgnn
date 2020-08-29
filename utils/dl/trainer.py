from utils.dl.report import Report
from utils.dl.metrics import BinaryMetric

import copy
class Trainer():
    def __init__(self, device, f_loss, f_pred):
        self.device = device
        self.f_loss = f_loss
        self.f_pred = f_pred

    def train_epoch(self, loader, model, optimizer, criterion):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(self.device)
            optimizer.zero_grad()
            output = model(data)
            loss = self.f_loss(criterion, output, data).to(self.device)
            loss.backward()
            loss_all += loss.item()
            optimizer.step()
        return loss_all / len(loader.dataset)

    def eval_epoch(self, loader, model, criterion):
        model.eval()
        y_true, y_score, y_pred = list(), list(), list()
        loss_all = 0
        for data in loader:
            data = data.to(self.device)
            output = model(data)

            loss = self.f_loss(criterion, output, data).to(self.device)
            pred = self.f_pred(output)

            loss_all += loss.item()
            y_true += data.y.cpu().numpy().tolist()
            y_score += output.detach().cpu().numpy().tolist()
            y_pred += pred.cpu().numpy().tolist()

        return loss_all / len(loader.dataset), y_true, y_score, y_pred

    def report_epoch(self, metrics, y_true, y_score, y_pred, **kwargs):
        r = Report()
        for m in metrics:
            r.add(m, y_true=y_true, y_score=y_score, y_pred=y_pred)
        r = {**r, **kwargs}
        return r

    def train(self, epochs, loaders, model, optimizer, criterion, metrics, best_model_crit: BinaryMetric, verbose=False):
        assert "train" in loaders and "valid" in loaders
        train_loader, valid_loader = loaders["train"], loaders["valid"]
        train_reports, valid_reports, test_reports = list(), list(), list()

        best_state, best_epoch = None, 0
        for epoch in range(0, epochs):
            _ = self.train_epoch(train_loader, model, optimizer, criterion)

            train_loss, train_true, train_score, train_pred = self.eval_epoch(train_loader, model, criterion)
            valid_loss, valid_true, valid_score, valid_pred = self.eval_epoch(valid_loader, model, criterion)
            train_rp = self.report_epoch(metrics, train_true, train_score, train_pred, loss=train_loss)
            valid_rp = self.report_epoch(metrics, valid_true, valid_score, valid_pred, loss=valid_loss)

            if epoch == 0 or valid_rp[best_model_crit.key()] > valid_reports[best_epoch][best_model_crit.key()]:
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            train_reports.append(train_rp)
            valid_reports.append(valid_rp)

            if "test" in loaders:
                test_loader = loaders["test"]
                test_loss, test_true, test_score, test_pred = self.eval_epoch(test_loader, model, criterion)
                test_rp = self.report_epoch(metrics, test_true, test_score, test_pred, loss=test_loss)
                test_reports.append(test_rp)

            if verbose:
                print('[ Epoch:{:03d}, Train Loss: {:.5f}, Valid Loss.{:.5f} ]' \
                      .format(epoch, train_loss, valid_loss))

        reports = {"train":train_reports, "valid":valid_reports, "test":test_reports if "test" in loaders else None}
        return {"epoch": best_epoch, "state": best_state}, reports