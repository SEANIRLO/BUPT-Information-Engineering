/*
 * Copyright (c) 2021 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.example.bburecog.slice;

import com.example.bburecog.utils.DoubleLineListItemFactory;
import com.example.bburecog.ResourceTable;

import ohos.aafwk.ability.AbilitySlice;
import ohos.aafwk.content.Intent;
import ohos.agp.components.Component;
import ohos.agp.components.DirectionalLayout;
import ohos.agp.components.Image;
import ohos.agp.components.ScrollView;
import ohos.agp.components.element.Element;
import ohos.agp.components.element.ElementScatter;
import ohos.bundle.AbilityInfo;
import ohos.global.configuration.Configuration;

/**
 * LoginAbilityAboutSlice
 */
public class LoginAbilityAboutSlice extends AbilitySlice {
    private static final int OVER_SCROLL_PERCENT = 20;
    private static final float OVER_SCROLL_RATE = 1.0f;
    private static final int REMAIN_VISIBLE_PERCENT = 20;
    private static final int ITEM_NUM = 2;
    public static final String RESULT_KEY = "com.example.bburecog.RESULT_KEY";

    private void initComponents(){
        initScrollView();
        initItems();
    }

    private void initAppBar() {
        DirectionalLayout backButton = (DirectionalLayout)
                findComponentById(ResourceTable.Id_appBar_backButton_touchTarget);
        Image backButtonImage = (Image) findComponentById(ResourceTable.Id_appBar_backButton);
        if (backButtonImage.getLayoutDirectionResolved() == Component.LayoutDirection.RTL) {
            Element buttonImage = ElementScatter.getInstance(this).parse(ResourceTable.Graphic_ic_back_mirror);
            backButtonImage.setImageElement(buttonImage);
        }
        backButton.setClickedListener(component -> onBackPressed());
    }

    private void initScrollView() {
        ScrollView scrollView = (ScrollView) findComponentById(ResourceTable.Id_aboutPageScrollView);
        scrollView.setReboundEffectParams(OVER_SCROLL_PERCENT, OVER_SCROLL_RATE, REMAIN_VISIBLE_PERCENT);
        scrollView.setReboundEffect(true);
    }

    private void initItems() {
        DoubleLineListItemFactory doubleLineListItemFactory = new DoubleLineListItemFactory(getContext());
        DirectionalLayout aboutPageList = (DirectionalLayout) findComponentById(ResourceTable.Id_aboutPageLowerPart);
        aboutPageList.removeAllComponents();
        // Add ITEM_NUM - 1 Components, manually hide the last component's divider
        for (int i = 0; i < ITEM_NUM - 1; i++) {
            aboutPageList.addComponent(doubleLineListItemFactory
                    .getDoubleLineList("BUPT SICE", "https://sice.bupt.edu.cn"));
        }
        DirectionalLayout lastItem = doubleLineListItemFactory
                .getDoubleLineList("China Mobile", "http://www.10086.cn");
        lastItem.findComponentById(ResourceTable.Id_divider).setVisibility(Component.INVISIBLE);
        aboutPageList.addComponent(lastItem);
    }

    @Override
    public void onStart(Intent intent) {
        super.onStart(intent);
        int orientation = getResourceManager().getConfiguration().direction;
        if (orientation == Configuration.DIRECTION_HORIZONTAL) {
            super.setUIContent(ResourceTable.Layout_login_ability_about_slice_landscape);
        } else {
            super.setUIContent(ResourceTable.Layout_login_ability_about_slice);
            initComponents();
        }
        initAppBar();
    }

    @Override
    protected void onOrientationChanged(AbilityInfo.DisplayOrientation displayOrientation) {
        if (displayOrientation == AbilityInfo.DisplayOrientation.LANDSCAPE) {
            setUIContent(ResourceTable.Layout_login_ability_about_slice_landscape);
        } else if (displayOrientation == AbilityInfo.DisplayOrientation.PORTRAIT) {
            setUIContent(ResourceTable.Layout_login_ability_about_slice);
            initScrollView();
            initItems();
        }
        initAppBar();
    }

    @Override
    public void onActive() {
        super.onActive();
    }

    @Override
    public void onForeground(Intent intent) {
        super.onForeground(intent);
    }
}
